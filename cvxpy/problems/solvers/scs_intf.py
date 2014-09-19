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
# Attempt to import SCS.
try:
    import scs
except ImportError:
    warnings.warn("The solver SCS could not be imported.")

class SCS(Solver):
    """An interface for the SCS solver.
    """
    def name(self):
        """The name of the solver.
        """
        return s.SCS

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
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

    def _shape_args(self, c, A, b, G, h, F, dims):
        """Returns the arguments that will be passed to the solver.
        """
        data = {"c": c}
        data["A"] = A
        data["b"] = b
        return (data, dims)

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
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        results = scs.solve(data, dims, **solver_opts)
        status = s.SOLVER_STATUS[s.SCS][results["info"]["status"]]
        if status in s.SOLUTION_PRESENT:
            primal_val = results["info"]["pobj"]
            value = primal_val + obj_offset
            eq_dual = results["y"][0:dims["f"]]
            ineq_dual = results["y"][dims["f"]:]
            return (status, value, results["x"], eq_dual, ineq_dual)
        else:
            return (status, None, None, None, None)
