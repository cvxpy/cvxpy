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
import sys

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.max_elemwise import max_elemwise
import numpy as np
if sys.version_info >= (3, 0):
    from functools import reduce

class min_elemwise(max_elemwise):
    """ Elementwise minimum. """
    # Returns the elementwise minimum.
    @max_elemwise.numpy_numeric
    def numeric(self, values):
        return reduce(np.minimum, values)

    def sign_from_args(self):
        """
        Reduces the list of argument signs according to the following rules:
            NEGATIVE, ANYTHING = NEGATIVE
            ZERO, UNKNOWN = NEGATIVE
            ZERO, ZERO = ZERO
            ZERO, POSITIVE = ZERO
            UNKNOWN, POSITIVE = UNKNOWN
            POSITIVE, POSITIVE = POSITIVE
        """
        arg_signs = [arg._dcp_attr.sign for arg in self.args]
        if u.Sign.NEGATIVE in arg_signs:
            min_sign = u.Sign.NEGATIVE
        elif u.Sign.ZERO in arg_signs:
            if u.Sign.UNKNOWN in arg_signs:
                min_sign = u.Sign.NEGATIVE
            else:
                min_sign = u.Sign.ZERO
        elif u.Sign.UNKNOWN in arg_signs:
            min_sign = u.Sign.UNKNOWN
        else:
            min_sign = u.Sign.POSITIVE

        return min_sign

    # Default curvature.
    def func_curvature(self):
        return u.Curvature.CONCAVE

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
        t = lu.create_var(size)
        constraints = []
        for obj in arg_objs:
            # Promote obj.
            if obj.size != size:
                obj = lu.promote(obj, size)
            constraints.append(lu.create_leq(t, obj))
        return (t, constraints)
