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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.abs import abs
import numpy as np
from .power import power
from fractions import Fraction

class huber(Elementwise):
    """The Huber function

    Huber(x, M) = 2M|x|-M^2 for |x| >= |M|
                  |x|^2 for |x| <= |M|
    M defaults to 1.

    Parameters
    ----------
    x : Expression
        A CVXPY expression.
    M : int/float or Parameter
    """
    def __init__(self, x, M=1):
        self.M = self.cast_to_const(M)
        super(huber, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the huber function applied elementwise to x.
        """
        x = values[0]
        output = np.zeros(x.shape)
        M = self.M.value
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if np.abs(x[row, col]) <= M:
                    output[row, col] = np.square(x[row, col])
                else:
                    output[row, col] = 2*M*np.abs(x[row, col]) - M**2
        return output

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arg, decreasing for negative.
        """
        return [u.monotonicity.SIGNED]

    def get_data(self):
        """Returns the parameter M.
        """
        return self.M

    def validate_arguments(self):
        """Checks that M >= 0 and is constant.
        """
        if not (self.M.is_positive() and self.M.is_constant() \
                and self.M.is_scalar()):
            raise ValueError("M must be a non-negative scalar constant.")

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        minimize n^2 + 2M|s|
        subject to s + n = x

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        M = data
        x = arg_objs[0]
        n = lu.create_var(size)
        s = lu.create_var(size)
        two = lu.create_const(2, (1, 1))
        if isinstance(M, Parameter):
            M = lu.create_param(M, (1, 1))
        else: # M is constant.
            M = lu.create_const(M.value, (1, 1))

        # n**2 + 2*M*|s|
        n2, constr_sq = power.graph_implementation([n], size, (2, (Fraction(1, 2), Fraction(1, 2))))
        abs_s, constr_abs = abs.graph_implementation([s], size)
        M_abs_s = lu.mul_expr(M, abs_s, size)
        obj = lu.sum_expr([n2, lu.mul_expr(two, M_abs_s, size)])
        # x == s + n
        constraints = constr_sq + constr_abs
        constraints.append(lu.create_eq(x, lu.sum_expr([n, s])))
        return (obj, constraints)
