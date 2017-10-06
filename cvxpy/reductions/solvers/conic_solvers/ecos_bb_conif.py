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

from .conic_solver import ConicSolver
import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import (
                                                    dims_to_solver_dict, ECOS)


class ECOS_BB(ECOS):
    """An interface for the ECOS BB solver.
    """

    # Solver capabilities.
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

    def apply(self, problem):
        data, inv_data = super(ECOS_BB, self).apply(problem)
        # Because the problem variable is single dimensional, every
        # boolean/integer index has length one.
        var = problem.variables()[0]
        data[s.BOOL_IDX] = [int(t[0]) for t in var.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in var.integer_idx]
        return data, inv_data

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import ecos
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])
        # Default verbose to false for BB wrapper.
        if 'mi_verbose' in solver_opts:
            mi_verbose = solver_opts['mi_verbose']
            del solver_opts['mi_verbose']
        else:
            mi_verbose = verbose
        solution = ecos.solve(data[s.C], data[s.G], data[s.H],
                              cones, data[s.A], data[s.B],
                              verbose=verbose,
                              mi_verbose=mi_verbose,
                              bool_vars_idx=data[s.BOOL_IDX],
                              int_vars_idx=data[s.INT_IDX],
                              **solver_opts)
        return solution
