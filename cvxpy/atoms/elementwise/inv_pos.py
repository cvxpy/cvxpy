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
from cvxpy.utilities import key_utils as ku
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from  cvxpy.atoms.quad_over_lin import quad_over_lin
from  cvxpy.atoms.affine.index import index
import numpy as np

class inv_pos(Elementwise):
    """ Elementwise 1/x, x >= 0 """
    def __init__(self, x):
        super(inv_pos, self).__init__(x)

    # Returns the elementwise inverse of x.
    @Elementwise.numpy_numeric
    def numeric(self, values):
        return 1.0/values[0]

    # Always positive.
    def sign_from_args(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.DECREASING]

    @staticmethod
    def graph_implementation(arg_objs, size, data):
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
        rows, cols = size
        x = arg_objs[0]
        t = lu.create_var(size)
        one = lu.create_const(1, (1, 1))
        constraints = []
        for i in xrange(rows):
            row_slc = ku.index_to_slice(i)
            for j in xrange(cols):
                col_slc = ku.index_to_slice(j)
                key = (row_slc, col_slc)
                xi, x_idx_constr = index.graph_implementation([x], (1, 1), key)
                obj, qol_constr = quad_over_lin.graph_implementation([one, xi],
                                                                     (1, 1),
                                                                     None)
                ti, t_idx_constr = index.graph_implementation([t], (1, 1), key)
                constraints += x_idx_constr + qol_constr + t_idx_constr
                constraints += [lu.create_leq(obj, ti),
                                lu.create_leq(lu.neg_expr(xi))]
        return (t, constraints)
