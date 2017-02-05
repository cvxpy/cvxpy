"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.settings as s
from cvxpy.problems.solvers.ecos_intf import ECOS
import numpy as np


class JuliaOpt(ECOS):
    """An interface for JuliaOpt solvers.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = True
    MIP_CAPABLE = True

    # Map of JuliaOpt status to CVXPY status.
    STATUS_MAP = {"Optimal": s.OPTIMAL,
                  "Optimal/Inaccurate": s.OPTIMAL_INACCURATE,
                  "Unbounded": s.UNBOUNDED,
                  "Unbounded/Inaccurate": s.UNBOUNDED_INACCURATE,
                  "Infeasible": s.INFEASIBLE,
                  "Infeasible/Inaccurate": s.INFEASIBLE_INACCURATE,
                  "Failure": s.SOLVER_ERROR,
                  "Indeterminate": s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.JULIA_OPT

    def import_solver(self):
        """Imports the solver.
        """
        import cmpb
        cmpb  # For flake8

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Should the previous solver result be used to warm_start?
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cmpb
        data = self.get_problem_data(objective,
                                     constraints,
                                     cached_data)
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        # Description of variables.
        var_len = data[s.C].size
        types = [cmpb.MPBFREECONE]
        lengths = [var_len]
        indices = np.arange(var_len)
        var_cones = cmpb.MPBCones(types, lengths, indices)
        # Description of cone constraints.
        constr_len = data[s.B].size
        dims = data[s.DIMS]
        types = []
        lengths = []
        if dims[s.EQ_DIM]:
            types.append(cmpb.MPBZEROCONE)
            lengths.append(dims[s.EQ_DIM])
        if dims[s.LEQ_DIM]:
            types.append(cmpb.MPBNONNEGCONE)
            lengths.append(dims[s.LEQ_DIM])
        for soc_len in dims[s.SOC_DIM]:
            types.append(cmpb.MPBSOC)
            lengths.append(soc_len)
        if dims[s.EXP_DIM]:
            types += [cmpb.MPBEXPPRIMAL]*dims[s.EXP_DIM]
            lengths += [3]*dims[s.EXP_DIM]
        indices = np.arange(constr_len)
        constr_cones = cmpb.MPBCones(types, lengths, indices)
        # Handle binary/integer variables.
        var_types = None
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            var_types = []
            for i in range(var_len):
                if i in data[s.BOOL_IDX]:
                    var_type = cmpb.MPBBINVAR
                elif i in data[s.INT_IDX]:
                    var_type = cmpb.MPBINTVAR
                else:
                    var_type = cmpb.MPBCONTVAR
                var_types.append(var_type)

        # Create solver object.
        Acoo = data[s.A].tocoo()
        model = cmpb.MPBModel(solver_opts["package"], solver_opts["solver_str"],
                              data[s.C], Acoo, data[s.B], constr_cones, var_cones, var_types)
        # Solve problem.
        model.optimize()
        # Collect results.
        results_dict = {}
        results_dict["status"] = model.status()
        results_dict["pobj"] = model.getproperty("objval")
        results_dict["x"] = model.getsolution()
        results_dict["y"] = model.getdual()
        return self.format_results(results_dict, data, cached_data)

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        dims = data[s.DIMS]
        new_results = {}
        status = self.STATUS_MAP[results_dict["status"]]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict["pobj"]
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict["x"]
            new_results[s.EQ_DUAL] = results_dict["y"][:dims[s.EQ_DIM]]
            new_results[s.INEQ_DUAL] = results_dict["y"][dims[s.EQ_DIM]:]

        return new_results
