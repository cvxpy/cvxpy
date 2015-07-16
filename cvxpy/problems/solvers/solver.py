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
        # If SDP, defaults to CVXOPT.
        elif constr_map[s.SDP]:
            return s.CVXOPT
        # Otherwise use ECOS.
        else:
            return s.ECOS

    def is_installed(self):
        """Is the solver installed?
        """
        try:
            self.import_solver()
            return True
        except ImportError:
            return False

    def validate_solver(self, constraints):
        """Raises an exception if the solver cannot solve the problem.

        Parameters
        ----------
        constraints: list
            The list of canonicalized constraints.
        """
        # Check the solver is installed.
        if not self.is_installed():
            raise SolverError("The solver %s is not installed." % self.name())
        # Check the solver can solve the problem.
        constr_map = SymData.filter_constraints(constraints)
        if ((constr_map[s.BOOL] or constr_map[s.INT]) \
            and not self.MIP_CAPABLE) or \
           (constr_map[s.SDP] and not self.SDP_CAPABLE) or \
           (constr_map[s.EXP] and not self.EXP_CAPABLE) or \
           (constr_map[s.SOC] and not self.SOCP_CAPABLE) or \
           (len(constraints) == 0 and self.name() in [s.SCS,
                                                      s.GLPK]):
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
        bool_idx, int_idx = self._noncvx_id_to_idx(data[s.DIMS],
                                                   sym_data.var_offsets,
                                                   sym_data.var_sizes)
        data[s.BOOL_IDX] = bool_idx
        data[s.INT_IDX] = int_idx
        return data

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
        return len(data[s.BOOL_IDX]) > 0 or len(data[s.BOOL_IDX]) > 0

    @staticmethod
    def _noncvx_id_to_idx(dims, var_offsets, var_sizes):
        """Converts the nonconvex constraint variable ids in dims into indices.

        Parameters
        ----------
        dims : dict
            The dimensions of the cones.
        var_offsets : dict
            A dict of variable id to horizontal offset.
        var_sizes : dict
            A dict of variable id to variable dimensions.

        Returns
        -------
        tuple
            A list of indices for the boolean variables and integer variables.
        """
        bool_idx = []
        int_idx = []
        for indices, constr_type in zip([bool_idx, int_idx],
                                        [s.BOOL_IDS, s.INT_IDS]):
            for var_id in dims[constr_type]:
                offset = var_offsets[var_id]
                size = var_sizes[var_id]
                for i in range(size[0]*size[1]):
                    indices.append(offset + i)
            del dims[constr_type]

        return bool_idx, int_idx
