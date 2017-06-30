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
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.problems.objective import Minimize
from cvxpy.expressions.attributes import is_quadratic
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.attributes import is_qp_constraint, are_arguments_affine
from cvxpy.problems.objective_attributes import is_qp_objective
from cvxpy.problems.problem import Problem
from cvxpy.reductions import InverseData


class QpMatrixStuffing(MatrixStuffing):
    """Fills in numeric values for this problem instance.
    """

    preconditions = {
        (Minimize, is_quadratic, True),
        (Constraint, are_arguments_affine, True),
        (Constraint, is_qp_constraint, True)
    }

    @staticmethod
    def postconditions(problem_type):
        return QpMatrixStuffing.preconditions.union({(Minimize, is_qp_objective, True)})

    def stuffed_objective(self, problem, inverse_data):
        # We need to copy the problem, because we are changing atoms in the expression tree
        problem_copy = Problem(Minimize(problem.objective.expr.tree_copy()),
                               [con.tree_copy() for con in problem.constraints])
        inverse_data_of_copy = InverseData(problem_copy)
        extractor = CoeffExtractor(inverse_data_of_copy)
        # extract to x.T * P * x + q.T * x, store r
        P, q, r = extractor.quad_form(problem_copy.objective.expr)

        # concatenate all variables in one vector
        x = cvxpy.Variable(inverse_data.x_length)
        new_obj = QuadForm(x, P) + q.T*x

        inverse_data.r = r
        return new_obj, x
