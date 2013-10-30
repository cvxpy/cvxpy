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

from .. import settings as s
from .. import utilities as u
from .. import interface as intf
from expression import Expression
import operator as op

class AffExpression(Expression):
    """ An affine expression. """
    # coefficients - a dict of {variable/Constant: [coefficients]}
    def __init__(self, coefficients, dcp_attr):
        self._coeffs = coefficients
        self._stored_dcp_attr = dcp_attr
        # Acts as a leaf node, so has no subexpressions.
        self.subexpressions = []

    # Multiplies the coefficient by the vectorized variable value
    # or simply adds it to the sum if constant.
    # Then combines columns into a single matrix.
    def numeric(self, values):
        # The interface for the variable values.
        interface = intf.DEFAULT_INTERFACE
        col_sums = self.size[1]*[0]
        for var,blocks in self.coefficients().items():
            # The vectorized value of the variable (or 1 if Constant).
            if var == s.CONSTANT:
                key_val = 1
            else:
                key_val = interface.const_to_matrix(var.value)
                key_val = interface.reshape(key_val, (var.size[0]*var.size[1],1))
            for i in xrange(self.size[1]):
                col_sums[i] += blocks[i]*key_val
        # Returns a sparse matrix.
        sparse_intf = intf.DEFAULT_SPARSE_INTERFACE
        result = sparse_intf.zeros(self.size[0], self.size[1])
        for i in xrange(self.size[1]):
            result[:,i] = col_sums[i]
        return result

    # Returns itself as an objective and any constraints
    # associated with the variables.
    def canonicalize(self):
        # Combines the contraints from the variables.
        constraints = []
        for var in self.coefficients().keys():
            if var != s.CONSTANT:
                constraints += var.canonicalize()[1]
        return (self, constraints)

    # The representation of the expression as a string.
    def name(self):
        return str(self.coefficients())

    # Return the stored curvature, sign, and shape of the
    # affine expression.
    def _dcp_attr(self):
        return self._stored_dcp_attr

    # Returns a list of the variables in the expression.
    def variables(self):
        variables = self.coefficients().keys()
        if s.CONSTANT in variables:
            variables.remove(s.CONSTANT)
        return variables

    # Returns a coefficient dict with all parameter expressions evaluated.
    def coefficients(self):
        return self._coeffs

    # Multiplies by a ones matrix to promote scalar coefficients.
    # Returns an updated coefficient dict.
    @staticmethod
    def promote(coeffs, shape):
        rows,cols = shape.size
        ones = intf.DEFAULT_NP_INTERFACE.ones(rows)
        new_coeffs = {}
        for var,blocks in coeffs.items():
            new_coeffs[var] = [ones*blocks[0] for i in range(cols)]
        return new_coeffs

    # Combines the dicts. Adds the blocks of common variables.
    # Multiplies all blocks by a ones vector if promotion occurs.
    def __add__(self, other):
        new_dcp_attr = self._dcp_attr() + other._dcp_attr()
        new_shape = new_dcp_attr.shape
        # Get the coefficients of the two expressions.
        self_coeffs = self.coefficients()
        other_coeffs = other.coefficients()
        if new_shape.size > self.size:
            self_coeffs = self.promote(self, new_shape)
        elif new_shape.size > other.size:
            other_coeffs = self.promote(other, new_shape)
        # Merge the dicts, summing common variables.
        new_coeffs = self_coeffs.copy()
        for var,blocks in other_coeffs.items():
            if var in new_coeffs:
                block_sum = []
                for (b1,b2) in zip(new_coeffs[var], other_coeffs[var]):
                    block_sum.append(b1 + b2)
                new_coeffs[var] = block_sum
            else:
                new_coeffs[var] = blocks
        return AffExpression(new_coeffs, new_dcp_attr)

    def __sub__(self, other):
        return self + -other

    # Distributes multiplications by left hand constant
    # across right hand terms.
    def __mul__(self, other):
        lh_blocks = self.coefficients()[s.CONSTANT]
        new_coeffs = {}
        for var,blocks in other.coefficients().items():
            block_product = []
            for (b1,b2) in zip(lh_blocks, blocks):
                prod = b1 * b2
                # Reduce to scalar if possible.
                if prod.size == (1,1):
                    prod = prod[0]
                block_product.append(prod)
            new_coeffs[var] = block_product
        return AffExpression(new_coeffs, self._dcp_attr() * other._dcp_attr())

    # Negates every parameter expression.
    def __neg__(self):
        new_coeffs = {}
        for var,blocks in self.coefficients().items():
            new_coeffs[var] = map(op.neg, blocks)
        return AffExpression(new_coeffs, self._dcp_attr())