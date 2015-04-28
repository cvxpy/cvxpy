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
