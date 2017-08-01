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

from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.matrix_stuffing import extract_mip_idx, MatrixStuffing
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.utilities import are_args_affine


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

    def accepts(self, problem):
        return (type(problem.objective) == Minimize
                and problem.objective.expr.is_affine()
                and not convex_attributes(problem.variables())
                and are_args_affine(problem.constraints))

    def stuffed_objective(self, problem, inverse_data):
        extractor = CoeffExtractor(inverse_data)
        # Extract to c.T * x, store r
        C, R = extractor.get_coeffs(problem.objective.expr)

        c = np.asarray(C.todense()).flatten()
        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(inverse_data.x_length, boolean=boolean, integer=integer)

        new_obj = c.T * x + 0

        inverse_data.r = R[0]
        return new_obj, x
