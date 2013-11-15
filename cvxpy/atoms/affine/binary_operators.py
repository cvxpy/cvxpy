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

from affine_atom import AffAtom
from ...utilities import coefficient_utils as cu
from ... import interface as intf
from ...expressions.constants import Constant
import operator as op
import numpy as np

class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators.
    """
    def __init__(self, lh_exp, rh_exp):
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        return ' '.join([self.args[0].name(),
                         self.OP_NAME,
                         self.args[1].name()])

    # Applies the binary operator to the values.
    def numeric(self, values):
        return reduce(self.OP_FUNC, values)

    # Returns the sign, curvature, and shape.
    def init_dcp_attr(self):
        self._dcp_attr = self.OP_FUNC(self.args[0]._dcp_attr,
                                      self.args[1]._dcp_attr)

    # Validate the dimensions.
    def validate_arguments(self):
        self.OP_FUNC(self.args[0].shape, self.args[1].shape)

class MulExpression(BinaryOperator):
    OP_NAME = "*"
    OP_FUNC = op.mul

    def func_coefficients(self):
        """Return the dict of Variable to coefficient for the product.
        """
        return cu.mul(self.args[0].coefficients(),
                      self.args[1].coefficients())

    # If left-hand side is non-constant, replace lh*rh with x, x.T == rh.T*lh.T.
    def graph_implementation(self, arg_objs):
        if not self.args[0].curvature.is_constant():
            x = Variable(*self.size)
            constraints = (x.T == arg_objs[1].T*arg_objs[0].T).canonicalize()[1]
            return (obj, constraints)
        else:
            return super(MulExpression, self).graph_implementation(arg_objs)
