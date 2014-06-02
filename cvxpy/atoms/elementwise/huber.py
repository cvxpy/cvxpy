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
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.square import square

def huber(x, M=1):
    """The Huber function

    Huber(x, M) = 2M|x|-M^2 for |x| >= |M|
                  |x|^2 for |x| <= |M|
    M defaults to 1.

    Parameters
    ----------
    x : Expression
        A CVXPY expression.
    M : int/float
    """
    # TODO require that M is positive?
    return square(M)*huber_pos(abs(x)/abs(M))

class huber_pos(Elementwise):
    """Elementwise Huber function for non-negative expressions and M=1.
    """
    def __init__(self, x):
        super(huber_pos, self).__init__(x)

    # Returns the huber function applied elementwise to x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if x[row, col] >= 1:
                    x[row, col] = 2*x[row, col] - 1
                else:
                    x[row, col] = x[row, col]**2

        return x

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.SIGNED]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

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
        x = arg_objs[0]
        w = lu.create_var(size)
        v = lu.create_var(size)
        two = lu.create_const(2, (1, 1))
        # w**2 + 2*v
        obj, constraints = square.graph_implementation([w], size)
        obj = lu.sum_expr([obj, lu.mul_expr(two, v, size)])
        # x <= w + v
        constraints.append(lu.create_leq(x, lu.sum_expr([w, v])))
        # v >= 0
        constraints.append(lu.create_geq(v))
        return (obj, constraints)
