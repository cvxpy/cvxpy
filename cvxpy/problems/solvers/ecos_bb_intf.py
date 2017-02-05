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


class ECOS_BB(ECOS):
    """An interface for the ECOS BB solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Exit flags from ECOS_BB
    # ECOS_BB found optimal solution.
    # MI_OPTIMAL_SOLN (ECOS_OPTIMAL)
    # ECOS_BB proved problem is infeasible.
    # MI_INFEASIBLE (ECOS_PINF)
    # ECOS_BB proved problem is unbounded.
    # MI_UNBOUNDED (ECOS_DINF)
    # ECOS_BB hit maximum iterations but a feasible solution was found and
    # the best seen feasible solution was returned.
    # MI_MAXITER_FEASIBLE_SOLN (ECOS_OPTIMAL + ECOS_INACC_OFFSET)
    # ECOS_BB hit maximum iterations without finding a feasible solution.
    # MI_MAXITER_NO_SOLN (ECOS_PINF + ECOS_INACC_OFFSET)
    # ECOS_BB hit maximum iterations without finding a feasible solution
    #   that was unbounded.
    # MI_MAXITER_UNBOUNDED (ECOS_DINF + ECOS_INACC_OFFSET)

    def name(self):
        """The name of the solver.
        """
        return s.ECOS_BB

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
        import ecos
        data = self.get_problem_data(objective, constraints, cached_data)
        # Default verbose to false for BB wrapper.
        mi_verbose = solver_opts.get('mi_verbose', False)
        results_dict = ecos.solve(data[s.C], data[s.G], data[s.H],
                                  data[s.DIMS], data[s.A], data[s.B],
                                  verbose=verbose,
                                  mi_verbose=mi_verbose,
                                  bool_vars_idx=data[s.BOOL_IDX],
                                  int_vars_idx=data[s.INT_IDX],
                                  **solver_opts)
        return self.format_results(results_dict, data, cached_data)
