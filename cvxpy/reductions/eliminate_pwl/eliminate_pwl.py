"""
Copyright 2017 Robin Verschueren

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

from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.problems.attributes import has_pwl_atoms
from cvxpy.problems.problem import Problem
from .atom_canonicalizers import CANON_METHODS as elim_pwl_methods


class EliminatePwl(Canonicalization):

    preconditions = {
        (Problem, has_pwl_atoms, True)
    }

    @staticmethod
    def postconditions(problem_type):
        return {(Problem, has_pwl_atoms, False)}

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot canonicalize pwl atoms away")
        return Canonicalization(elim_pwl_methods).apply(problem)
