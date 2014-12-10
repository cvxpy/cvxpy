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
from cvxpy.problems.solvers.scs_intf import SCS
import cvxpy.problems.iterative as iterative
import cvxpy.lin_ops.tree_mat as tree_mat
import scs_mat_free

class SCS_MAT_FREE(SCS):
    """An interface for the SCS solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.SCS_MAT_FREE

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
        return ([], [], [])

    def get_problem_data(self, objective, constraints, cached_data):
        """Returns the argument for the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        tuple
            (solver args tuple, offset)
        """
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        matrix_data = self.get_matrix_data(objective, constraints,
                                           cached_data)
        c, offset = matrix_data.get_objective()
        all_ineq = sym_data.constr_map[s.EQ] + sym_data.constr_map[s.LEQ]
        A_rows = sum(constr.size[0]*constr.size[1] for constr in all_ineq)
        b = -iterative.constr_mul(all_ineq, {}, A_rows, False)
        data = {"c": c}
        #data["A"] = matrix_data.get_eq_constr()[0]
        data["A"] = self.matrix_intf().zeros(A_rows, sym_data.x_length)
        data["b"] = b
        # Remove constants from expressions.
        constraints = tree_mat.prune_constants(all_ineq)
        Amul, ATmul = iterative.get_mul_funcs(sym_data, constraints)
        data["Amul"] = Amul
        data["ATmul"] = ATmul
        return (data, sym_data.dims), offset

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
        (data, dims), obj_offset = self.get_problem_data(objective,
                                                         constraints,
                                                         cached_data)
        # Always use indirect method.
        solver_opts["use_indirect"] = True
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        results_dict = scs_mat_free.solve(data, dims, **solver_opts)
        return self.format_results(results_dict, dims, obj_offset)

