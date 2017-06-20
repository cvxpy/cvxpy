"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

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

import cvxpy
from cvxpy.atoms import QuadForm
from cvxpy.problems.problem import Problem
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.attributes import is_quadratic, is_affine
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.problem_analyzer import ProblemAnalyzer
from cvxpy.problems.attributes import (has_affine_equality_constraints,
                                       has_affine_inequality_constraints)
from cvxpy.constraints.attributes import is_qp_constraint
from cvxpy.constraints import NonPos, Zero
from cvxpy.problems.objective_attributes import is_qp_objective


class QpMatrixStuffing(MatrixStuffing):
    """Fills in numeric values for this problem instance.
    """

    preconditions = {
        (Minimize, is_quadratic, True),
        (Constraint, is_affine, True),
        (Constraint, is_qp_constraint, True)
    }

    def accepts(self, problem):
        return ProblemAnalyzer(problem).matches(self.preconditions)

    def postconditions(self, problem):
        problem_type = ProblemAnalyzer(problem).type
        post_conditions = [(Minimize, is_qp_objective, True)]
        if (Problem, has_affine_inequality_constraints, True) in problem_type:
            post_conditions += [(NonPos, is_affine, True)]
        if (Problem, has_affine_equality_constraints, True) in problem_type:
            post_conditions += [(Zero, is_affine, True)]
        return post_conditions

    def stuffed_objective(self, problem, inverse_data):
        extractor = CoeffExtractor(inverse_data)
        # extract to x.T * P * x + q.T * x, store r
        (P, q, r) = extractor.quad_form(problem)

        # concatenate all variables in one vector
        x = cvxpy.Variable(inverse_data.x_length)
        new_obj = QuadForm(x, P) + q.T*x

        inverse_data.r = r
        return new_obj, x
