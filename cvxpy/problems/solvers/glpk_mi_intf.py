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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.glpk_intf import GLPK


class GLPK_MI(GLPK):
    """An interface for the GLPK MI solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Map of GLPK MIP status to CVXPY status.
    STATUS_MAP = {'optimal': s.OPTIMAL,
                  'feasible': s.OPTIMAL_INACCURATE,
                  'undefined': s.SOLVER_ERROR,
                  'invalid formulation': s.SOLVER_ERROR,
                  'infeasible problem': s.INFEASIBLE,
                  'LP relaxation is primal infeasible': s.INFEASIBLE,
                  'LP relaxation is dual infeasible': s.UNBOUNDED,
                  'unknown': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.GLPK_MI

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
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cvxopt
        import cvxopt.glpk
        data = self.get_problem_data(objective, constraints, cached_data)
        # Save original cvxopt solver options.
        old_options = cvxopt.glpk.options.copy()
        # Silence cvxopt if verbose is False.
        if verbose:
            cvxopt.glpk.options["msg_lev"] = "GLP_MSG_ON"
        else:
            cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

        # Apply any user-specific options.
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.glpk.options[key] = value

        try:
            results_tup = cvxopt.glpk.ilp(data[s.C],
                                          data[s.G],
                                          data[s.H],
                                          data[s.A],
                                          data[s.B],
                                          set(data[s.INT_IDX]),
                                          set(data[s.BOOL_IDX]))
            results_dict = {}
            results_dict["status"] = results_tup[0]
            results_dict["x"] = results_tup[1]
        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {"status": "unknown"}

        # Restore original cvxopt solver options.
        self._restore_solver_options(old_options)
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
        new_results = {}
        status = self.STATUS_MAP[results_dict['status']]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            # No dual variables.
            new_results[s.PRIMAL] = intf.cvxopt2dense(results_dict['x'])
            primal_val = (data[s.C].T*results_dict['x'])[0]
            new_results[s.VALUE] = primal_val + data[s.OFFSET]

        return new_results

    def _restore_solver_options(self, old_options):
        import cvxopt.glpk
        for key, value in list(cvxopt.glpk.options.items()):
            if key in old_options:
                cvxopt.glpk.options[key] = old_options[key]
            else:
                del cvxopt.glpk.options[key]
