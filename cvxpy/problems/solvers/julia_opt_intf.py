"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
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
    STATUS_MAP = {"Solved": s.OPTIMAL,
                  "Solved/Inaccurate": s.OPTIMAL_INACCURATE,
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
        types = np.ndarray([cmpb.MPBFREECONE]*var_len)
        lengths = np.ones(var_len)
        indices = np.arange(var_len)
        varcones = cmpb.MPBCones(types, lengths, indices)
        # Description of cone constraints.
        constr_len = data[s.B].size
        dims = data[s.DIMS]
        types = [cmpb.MPBZEROCONE]*dims[s.EQ_DIM]
        types += [cmpb.MPBNONPOSCONE]*dims[s.LEQ_DIM]
        lengths = [1]*(dims[s.EQ_DIM] + dims[s.LEQ_DIM])
        for soc_len in dims[s.SOC_DIM]:
            types.append(cmpb.MPBSOC)
            lengths.append(soc_len)
        types += [cmpb.MPBEXPPRIMAL]*dims[s.EXP_DIM]
        lengths += [3]*dims[s.EXP_DIM]
        indices = np.arange(constr_len)
        varcones = cmpb.MPBCones(types, lengths, indices)
        # Create solver object.
        model = MPBModel(solver_opts["package"], solver_opts["solver_str"],
                 data[s.C], data[s.A], data[s.B], constrcones, varcones)
        results_dict = cmpb.solve(scs_args, data[s.DIMS], **solver_opts)
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
        status = self.STATUS_MAP[results_dict["info"]["status"]]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict["info"]["pobj"]
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict["x"]
            new_results[s.EQ_DUAL] = results_dict["y"][:dims[s.EQ_DIM]]
            new_results[s.INEQ_DUAL] = results_dict["y"][dims[s.EQ_DIM]:]

        return new_results
