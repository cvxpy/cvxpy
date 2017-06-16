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
from cvxpy.reductions.qp2quad_form.atom_canonicalizers import CANON_METHODS as qp_canon_methods
from cvxpy.problems.objective import Minimize
from cvxpy.constraints import NonPos, Zero
from cvxpy.problems.problem_analyzer import ProblemAnalyzer
from cvxpy.problems.problem import Problem
from cvxpy.expressions.attributes import is_affine, is_quadratic, is_qpwa, is_pwl
from cvxpy.problems.attributes import (has_affine_equality_constraints,
                                       has_affine_inequality_constraints)


class Qp2SymbolicQp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine expressions
    and symbolic quadratic forms.
    """

    preconditions = {
                        (Minimize, is_qpwa, True),
                        (NonPos, is_pwl, True),
                        (Zero, is_affine, True)
                    }

    def accepts(self, problem):
        return ProblemAnalyzer(problem).check(self.preconditions)

    def get_postconditions(self, problem):
        problem_type = ProblemAnalyzer(problem).type
        post_conditions = [(Minimize, is_quadratic, True)]
        if (Problem, has_affine_inequality_constraints, True) in (problem_type):
            post_conditions += [(NonPos, is_affine, True)]
        if (Problem, has_affine_equality_constraints, True) in problem_type:
            post_conditions += [(Zero, is_affine, True)]
        return post_conditions

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        return Canonicalization(qp_canon_methods).apply(problem)
