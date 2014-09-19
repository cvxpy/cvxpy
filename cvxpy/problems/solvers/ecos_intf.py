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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
import ecos

class ECOS(Solver):
    """An interface for the ECOS solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.ECOS

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTERFACE

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTERFACE

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
        return (constr_map[s.EQ], constr_map[s.LEQ], [])

    def _shape_args(self, c, A, b, G, h, F, dims):
        """Returns the arguments that will be passed to the solver.
        """
        return (c, G, h, dims, A, b)

    def solve(self, objective, constraints, cached_data, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        prob_data = self.get_problem_data(objective, constraints, cached_data)
        obj_offset = prob_data[1]
        results = ecos.solve(*prob_data[0], verbose=verbose, **solver_opts)
        status = s.SOLVER_STATUS[s.ECOS][results['info']['exitFlag']]
        if status in s.SOLUTION_PRESENT:
            primal_val = results['info']['pcost']
            value = primal_val + obj_offset
            return (status, value,
                    results['x'], results['y'], results['z'])
        else:
            return (status, None, None, None, None)
