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

from cvxpy.atoms.atom import Atom
import cvxpy.interface as intf
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.sum_entries import sum_entries
import numpy as np

class sum_largest(Atom):
    """Sum of the largest k values in the matrix X.
    """
    def __init__(self, x, k):
        self.k = k
        super(sum_largest, self).__init__(x)

    def validate_arguments(self):
        """Verify that k is a positive integer.
        """
        if int(self.k) != self.k or self.k <= 0:
            raise ValueError("Second argument must be a positive integer.")

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the k largest entries of the matrix.
        """
        value = intf.from_2D_to_1D(values[0].flatten().T)
        indices = np.argsort(-value)[:int(self.k)]
        return value[indices].sum()

    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Same as the argument.
        """
        return self.args[0]._dcp_attr.sign

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Always increasing.
        """
        return [u.monotonicity.INCREASING]

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

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
        # min sum_entries(t) + kq
        # s.t. x <= t + q
        #      0 <= t
        x = arg_objs[0]
        k = lu.create_const(data[0], (1, 1))
        q = lu.create_var((1, 1))
        t = lu.create_var(x.size)
        sum_t, constr = sum_entries.graph_implementation([t], (1, 1))
        obj = lu.sum_expr([sum_t, lu.mul_expr(k, q, (1, 1))])
        prom_q = lu.promote(q, x.size)
        constr.append( lu.create_leq(x, lu.sum_expr([t, prom_q])) )
        constr.append( lu.create_geq(t) )
        return (obj, constr)
