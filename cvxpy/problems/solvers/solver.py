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
import cvxpy.settings as s
from cvxpy.problems.problem_data.matrix_data import MatrixData
from cvxpy.problems.problem_data.sym_data import SymData


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
    def import_solver(self):
        """Imports the solver.
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
        pass

    def is_installed(self):
        """Is the solver installed?
        """
        try:
            self.import_solver()
            return True
        except ImportError:
            return False

    def get_sym_data(self, objective, constraints, cached_data):
        """Returns the symbolic data for the problem.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized constraints.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        SymData
            The symbolic data for the problem.
        """
        prob_data = cached_data[self.name()]
        if prob_data.sym_data is None:
            prob_data.sym_data = SymData(objective, constraints, self)
        return prob_data.sym_data

    def get_matrix_data(self, objective, constraints, cached_data):
        """Returns the numeric data for the problem.

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
        SymData
            The symbolic data for the problem.
        """
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        prob_data = cached_data[self.name()]
        if prob_data.matrix_data is None:
            prob_data.matrix_data = MatrixData(sym_data,
                                               self.matrix_intf(),
                                               self.vec_intf(),
                                               self,
                                               self.nonlin_constr())
        return prob_data.matrix_data

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
        dict
            The arguments needed for the solver.
        """
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        matrix_data = self.get_matrix_data(objective, constraints,
                                           cached_data)
        data = {}
        data[s.C], data[s.OFFSET] = matrix_data.get_objective()
        data[s.A], data[s.B] = matrix_data.get_eq_constr()
        data[s.G], data[s.H] = matrix_data.get_ineq_constr()
        data[s.F] = matrix_data.get_nonlin_constr()
        data[s.DIMS] = sym_data.dims.copy()
        return data

    def nonlin_constr(self):
        """Returns whether nonlinear constraints are needed.
        """
        return False

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @staticmethod
    def is_mip(data):
        """Is the problem a mixed integer program?
        """
        return len(data[s.BOOL_IDX]) > 0 or len(data[s.INT_IDX]) > 0
