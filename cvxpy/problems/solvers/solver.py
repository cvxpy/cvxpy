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
from cvxpy.error import SolverError
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

    @abc.abstractmethod
    def sdp_capable(self):
        """Can the solver handle SDPs?
        """
        pass

    @abc.abstractmethod
    def exp_capable(self):
        """Can the solver handle the exponential cone?
        """
        pass

    @abc.abstractmethod
    def mip_capable(self):
        """Can the solver handle boolean or integer variables?
        """
        pass

    @staticmethod
    def choose_solver(constraints):
        """Determines the appropriate solver.

        Parameters
        ----------
        constraints: list
            The list of canonicalized constraints.

        Returns
        -------
        str
            The solver that will be used.
        """
        constr_map = SymData.filter_constraints(constraints)
        # If no constraints, use ECOS.
        if len(constraints) == 0:
            return s.ECOS
        # If mixed integer constraints, use ECOS_BB.
        elif constr_map[s.BOOL] or constr_map[s.INT]:
            return s.ECOS_BB
        # If SDP or EXP, defaults to CVXOPT.
        elif constr_map[s.SDP] or constr_map[s.EXP]:
            return s.CVXOPT
        # Otherwise use ECOS.
        else:
            return s.ECOS

    def validate_solver(self, constraints):
        """Raises an exception if the solver cannot solve the problem.

        Parameters
        ----------
        constraints: list
            The list of canonicalized constraints.
        """
        constr_map = SymData.filter_constraints(constraints)
        if ((constr_map[s.BOOL] or constr_map[s.INT]) \
            and not self.mip_capable()) or \
           (constr_map[s.SDP] and not self.sdp_capable()) or \
           (constr_map[s.EXP] and not self.exp_capable()) or \
           (len(constraints) == 0 and self.name() == s.SCS):
            raise SolverError(
                "The solver %s cannot solve the problem." % self.name()
            )

    def validate_cache(self, objective, constraints, cached_data):
        """Clears the cache if the objective or constraints changed.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        """
        prob_data = cached_data[self.name()]
        if prob_data.sym_data is not None and \
           (objective != prob_data.sym_data.objective or \
            constraints != prob_data.sym_data.constraints):
            prob_data.sym_data = None
            prob_data.matrix_data = None

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
        self.validate_cache(objective, constraints, cached_data)
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
                                               self)
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
        tuple
            (solver args tuple, offset)
        """
        sym_data = self.get_sym_data(objective, constraints, cached_data)
        matrix_data = self.get_matrix_data(objective, constraints,
                                           cached_data)
        c, offset = matrix_data.get_objective()
        A, b = matrix_data.get_eq_constr()
        G, h = matrix_data.get_ineq_constr()
        F = matrix_data.get_nonlin_constr()
        args = self._shape_args(c, A, b, G, h, F, sym_data.dims)
        return (args, offset)

    @abc.abstractmethod
    def _shape_args(self, c, A, b, G, h, F, dims):
        """Returns the arguments that will be passed to the solver.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass
