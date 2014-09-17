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

import abc

class Solver(object):
    """Generic interface for a solver.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def name(self):
        """The name of the solver.
        """
        pass

    @abc.abstractmethod
    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        pass

    @abc.abstractmethod
    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        pass

    @abc.abstractmethod
    def _get_constr(self, constr_map):
        """Extracts the equality, inequaliyt, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        pass

    def get_problem_data(self, cached_data, objective,
                         constr_map, dims,
                         var_offsets, x_length):
        """Returns the problem data for the call to the solver.

        Parameters
        ----------
        cached_data : dict
            A map of solver name to cached problem data.
        objective : LinOp
            The canonicalized objective.
        constr_map : dict
            A dict of the canonicalized constraints.
        dims : dict
            A dict with information about the types of constraints.
        var_offsets : dict
            A dict mapping variable id to offset in the stacked variable x.
        x_length : int
            The height of x.
        Returns
        -------
        tuple
            (solver args tuple, offset)
        """
        if self.name() not in cached_data:
            eq_constr, ineq_constr, nonlin = self._get_constr(constr_map)
            cached_data[self.name()] = ProblemData(var_offsets,
                                                   x_length,
                                                   objective,
                                                   eq_constr,
                                                   ineq_constr,
                                                   nonlin,
                                                   self.matrix_intf(),
                                                   self.vec_intf())
        problem_data = cached_data[self.name()]
        c, offset = problem_data.get_objective()
        A, b = problem_data.get_eq_constr()
        G, h = problem_data.get_ineq_constr()
        F = problem_data.get_nonlin_constr()
        args = self._shape_args(c, A, b, G, h, F, dims)
        return (args, offset)

    @abc.abstractmethod
    def _shape_args(c, A, b, G, h, F, dims):
        """Returns the arguments that will be passed to the solver.
        """
        pass

    def solve(self, cached_data, objective,
              constr_map, dims,
              var_offsets, x_length):
