"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy.expressions.variable import upper_tri_to_full
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize
from cvxpy.reductions import Solution
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers import (CANON_METHODS as
                                                           cone_canon_methods)
import numpy as np


class Dcp2Cone(Canonicalization):
    def accepts(self, problem):
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to cone program")
        return Canonicalization(cone_canon_methods).apply(problem)

    def invert(self, solution, inverse_data):
        pvars = {}
        for id, var in inverse_data.id2var.items():
            # Need to map from constrained to symmetric variable.
            if id in solution.primal_vars:
                if var.is_symmetric():
                    n = var.shape[0]
                    value = np.zeros(var.shape)
                    value[:n*(n+1)//2] = solution.primal_vars[id]
                    pvars[id] = value + value.T
                else:
                    pvars[id] = solution.primal_vars[id]
        dvars = {orig_id: solution.dual_vars[id]
                 for orig_id, id in inverse_data.cons_id_map.items()}
        return Solution(solution.status, solution.opt_val, pvars, dvars)
