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

    def sdp_capable(self):
        """Can the solver handle SDPs?
        """
        return True

    def exp_capable(self):
        """Can the solver handle the exponential cone?
        """
        return True

    def mip_capable(self):
        """Can the solver handle boolean or integer variables?
        """
        return False

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
        results_dict = ecos.solve(*prob_data[0], verbose=verbose,
                                  **solver_opts)
        return self.format_results(results_dict, None, obj_offset)

    def format_results(self, results_dict, dims, obj_offset=0):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        dims : dict
            The cone dimensions in the canonicalized problem.
        obj_offset : float, optional
            The constant term in the objective.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        status = s.SOLVER_STATUS[s.ECOS][results_dict['info']['exitFlag']]
        new_results[s.STATUS] = status
        new_results[s.SOLVE_TIME] = results_dict['info']["timing"]['runtime']
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['info']['pcost']
            new_results[s.VALUE] = primal_val + obj_offset
            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.EQ_DUAL] = results_dict['y']
            new_results[s.INEQ_DUAL] = results_dict['z']

        return new_results
