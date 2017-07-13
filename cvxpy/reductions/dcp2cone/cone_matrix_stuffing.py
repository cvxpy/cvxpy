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

import numpy as np

from cvxpy.expressions.variables import Variable
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.problems.objective import Minimize
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.attributes import is_affine
from cvxpy.constraints.attributes import (exists,
                                          is_cone_constraint,
                                          are_arguments_affine,
                                          is_stuffed_cone_constraint)
from cvxpy.problems.objective_attributes import is_cone_objective
from cvxpy.problems.problem import Problem
from cvxpy.problems.attributes import is_minimization


class ConeMatrixStuffing(MatrixStuffing):
    """Construct matrices for linear cone problems.

    Linear cone problems are assumed to have a linear objective and cone
    constraints which may have zero or more arguments, all of which must be
    affine.

    minimize   c'x
    subject to cone_constr1(A_1*x + b_1, ...)
               ...
               cone_constrK(A_i*x + b_i, ...)
    """

    preconditions = {
        (Problem, is_minimization, True),
        (Minimize, is_affine, True),
        (Constraint, is_cone_constraint, True),
        (Constraint, are_arguments_affine, True)
    }

    @staticmethod
    def postconditions(problem_type):
        post = set(cond for cond in problem_type if cond[1] == exists)
        post = post.union(ConeMatrixStuffing.preconditions)
        post = post.union({(Constraint, is_stuffed_cone_constraint, True)})
        return post.union({(Minimize, is_cone_objective, True)})

    def stuffed_objective(self, problem, inverse_data):
        extractor = CoeffExtractor(inverse_data)
        # Extract to c.T * x, store r
        C, R = extractor.get_coeffs(problem.objective.expr)

        c = np.asarray(C.todense()).flatten()
        x = Variable(inverse_data.x_length)
        new_obj = c.T * x + 0

        inverse_data.r = R[0]
        return new_obj, x
