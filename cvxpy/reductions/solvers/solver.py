"""
Copyright 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc
from cvxpy.constraints import SOC, ExpCone, NonPos, PSD, Zero
from cvxpy.reductions.reduction import Reduction
import cvxpy.settings as s


class ConeDims(object):
    """Summary of cone dimensions present in constraints.

    Constraints must be formatted as dictionary that maps from
    constraint type to a list of constraints of that type.

    Attributes
    ----------
    zero : int
        The dimension of the zero cone.
    nonpos : int
        The dimension of the non-positive cone.
    exp : int
        The number of 3-dimensional exponential cones
    soc : list of int
        A list of the second-order cone dimensions.
    psd : list of int
        A list of the positive semidefinite cone dimensions, where the
        dimension of the PSD cone of k by k matrices is k.
    """
    def __init__(self, constr_map):
        self.zero = int(sum(c.size for c in constr_map[Zero]))
        self.nonpos = int(sum(c.size for c in constr_map[NonPos]))
        self.exp = int(sum(c.num_cones() for c in constr_map[ExpCone]))
        self.soc = [int(dim) for c in constr_map[SOC] for dim in c.cone_sizes()]
        self.psd = [int(c.shape[0]) for c in constr_map[PSD]]

    def __repr__(self):
        return "(zero: {0}, nonpos: {1}, exp: {2}, soc: {3}, psd: {4})".format(
            self.zero, self.nonpos, self.exp, self.soc, self.psd)

    def __str__(self):
        """String representation.
        """
        return ("%i equalities, %i inequalities, %i exponential cones, \n"
                "SOC constraints: %s, PSD constraints: %s.") % (self.zero,
                                                                self.nonpos,
                                                                self.exp,
                                                                self.soc,
                                                                self.psd)

    def __getitem__(self, key):
        if key == s.EQ_DIM:
            return self.zero
        elif key == s.LEQ_DIM:
            return self.nonpos
        elif key == s.EXP_DIM:
            return self.exp
        elif key == s.SOC_DIM:
            return self.soc
        elif key == s.PSD_DIM:
            return self.psd
        else:
            raise KeyError(key)


class Solver(Reduction):
    """Generic interface for a solver that uses reduction semantics
    """
    # The key that maps to ConeDims in the data returned by apply().
    DIMS = "dims"

    __metaclass__ = abc.ABCMeta

    # Solver capabilities.
    MIP_CAPABLE = False

    # Keys for inverse data.
    VAR_ID = 'var_id'
    DUAL_VAR_ID = 'dual_var_id'
    EQ_CONSTR = 'eq_constr'
    NEQ_CONSTR = 'other_constr'

    @abc.abstractmethod
    def name(self):
        """The name of the solver.
        """
        return NotImplemented

    @abc.abstractmethod
    def import_solver(self):
        """Imports the solver.
        """
        return NotImplemented

    def is_installed(self):
        """Is the solver installed?
        """
        try:
            self.import_solver()
            return True
        except ImportError:
            return False

    @abc.abstractmethod
    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        """Solve a problem represented by data returned from apply.
        """
        return NotImplemented

    def solve(self, problem, warm_start, verbose, solver_opts):
        """Solve the problem and return a Solution object.
        """
        data, inv_data = self.apply(problem)
        solution = self.solve_via_data(data, warm_start, verbose, solver_opts)
        return self.invert(solution, inv_data)
