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

from .. import interface as intf
from .. import settings as s
from .. import utilities as u
from ..utilities import key_utils as ku
from expression import Expression, cast_other
import types
import operator as op

class AffExpression(Expression):
    """ An affine expression. """
    # coefficients - a dict of {variable/Constant: [coefficients]}.
    # dcp_attr - the curvature, sign, and shape of the expression.
    def __init__(self, coefficients, dcp_attr):
        self._coeffs = self.format_coeffs(coefficients)
        self._stored_dcp_attr = dcp_attr
        # Acts as a leaf node, so has no subexpressions.
        self.subexpressions = []

    # Reduces the coefficients to scalars if possible.
    def format_coeffs(self, coefficients):
        for var_id,blocks in coefficients.items():
            for i,block in enumerate(blocks):
                if intf.is_scalar(block):
                    blocks[i] = intf.scalar_value(block)
        return coefficients

    # Multiplies the coefficient by the vectorized variable value
    # or simply adds it to the sum if constant.
    # Then combines columns into a single matrix.
    def numeric(self, values):
        # The interface for the variable values.
        interface = intf.DEFAULT_INTERFACE
        col_sums = self.size[1]*[0]
        for key,blocks in self.coefficients().items():
            # The vectorized value of the variable (or 1 if Constant).
            if key is s.CONSTANT:
                key_val = 1
            else:
                # Break if variable has no value.
                if key.value is None:
                    return None
                key_val = interface.const_to_matrix(var.value)
                key_val = interface.reshape(key_val, (var.size[0]*var.size[1],1))
            for i in xrange(self.size[1]):
                col_sums[i] += blocks[i]*key_val
        return self.merge_cols(col_sums)

    # Utility method to merge column blocks into a single matrix.
    # Returns a sparse matrix.
    def merge_cols(self, blocks):
        # Check for scalars.
        if self.size == (1,1):
            return blocks[0]
        interface = intf.DEFAULT_SPARSE_INTERFACE
        result = interface.zeros(self.size[0], self.size[1])
        for i in xrange(self.size[1]):
            result[:,i] = blocks[i]
        return result

    # Returns itself as an objective and any constraints
    # associated with the variables.
    def canonicalize(self):
        # Combines the contraints from the variables.
        constraints = []
        for var in self.variables():
            constraints += var.canonicalize()[1]
        return (self, constraints)

    # The representation of the expression as a string.
    def name(self):
        return str(self.coefficients())

    # Return the stored curvature, sign, and shape of the
    # affine expression.
    def _dcp_attr(self):
        return self._stored_dcp_attr

    # Returns a dict of variable id to variable object
    # with entries for all the variables in the expression.
    def variables(self):
        return [var for var in self.coefficients().keys() if var is not s.CONSTANT]

    # Returns a coefficient dict with all parameter expressions evaluated.
    def coefficients(self):
        return self._coeffs

    """ Indexing/Slicing """
    # Indexes/slices into the coefficients of the argument.  
    def __getitem__(self, key):
        # Format and validate the key.
        key = ku.validate_key(key, self.shape)
        new_coeffs = {}
        for var_id,blocks in self.coefficients().items():
            new_blocks = []
            # Indexes into the rows of the coefficients.
            for block in blocks[key[1]]:
                block_val = intf.index(block, (key[0],slice(None,None,None)))
                new_blocks.append(block_val)
            new_coeffs[var_id] = new_blocks

        return AffExpression(new_coeffs, self._dcp_attr()[key])

    """ Arithmetic operators """
    # Expression OP AffExpression is handled by the 
    # Expression class.
    @staticmethod
    def cast_to_const(expr):
        if isinstance(expr, AffExpression):
            return expr
        if isinstance(expr, Expression):
            return NotImplemented
        else:
            return types.constant()(expr)
                
    # Multiplies by a ones matrix to promote scalar coefficients.
    # Returns an updated coefficient dict.
    @staticmethod
    def promote(coeffs, shape):
        rows,cols = shape.size
        ones = intf.DEFAULT_SPARSE_INTERFACE.ones(rows, 1)
        new_coeffs = {}
        for key,blocks in coeffs.items():
            new_coeffs[key] = [ones*blocks[0] for i in range(cols)]
        return new_coeffs

    # Combines the dicts. Adds the blocks of common variables.
    # Multiplies all blocks by a ones vector if promotion occurs.
    @cast_other
    def __add__(self, other):
        new_dcp_attr = self._dcp_attr() + other._dcp_attr()
        new_shape = new_dcp_attr.shape
        # Get the coefficients of the two expressions.
        # Promote the coefficients if necessary.
        self_coeffs = self.coefficients()
        other_coeffs = other.coefficients()
        if new_shape.size > self.size:
            self_coeffs = self.promote(self_coeffs, new_shape)
        elif new_shape.size > other.size:
            other_coeffs = self.promote(other_coeffs, new_shape)
        # Merge the dicts, summing common variables.
        new_coeffs = self_coeffs.copy()
        for var_id,blocks in other_coeffs.items():
            if var_id in new_coeffs:
                block_sum = []
                for (b1,b2) in zip(new_coeffs[var_id], other_coeffs[var_id]):
                    block_sum.append(b1 + b2)
                new_coeffs[var_id] = block_sum
            else:
                new_coeffs[var_id] = blocks
        return AffExpression(new_coeffs, new_dcp_attr)

    # Called for Number + AffExpression.
    @cast_other
    def __radd__(self, other):
        return other + self

    @cast_other
    def __sub__(self, other):
        return self + -other

    # Called for Number - AffExpression.
    @cast_other
    def __rsub__(self, other):
        return other - self

    # Distributes multiplications by left hand constant
    # across right hand terms.
    @cast_other
    def __mul__(self, other):
        # Get new attributes and verify that dimensions are valid.
        dcp_attr = self._dcp_attr() * other._dcp_attr()
        # Multiply all coefficients by the constant.
        lh_blocks = self.coefficients()[s.CONSTANT]
        constant_term = self.merge_cols(lh_blocks)
        new_coeffs = {}
        for var_id,blocks in other.coefficients().items():
            block_product = []
            for block in blocks:
                block_product.append(constant_term * block)
            new_coeffs[var_id] = block_product
        return AffExpression(new_coeffs, dcp_attr)

    # Called for Number * AffExpression.
    @cast_other
    def __rmul__(self, other):
        return other * self

    # Negates every parameter expression.
    def __neg__(self):
        new_coeffs = {}
        for var_id,blocks in self.coefficients().items():
            new_coeffs[var_id] = map(op.neg, blocks)
        return AffExpression(new_coeffs, -self._dcp_attr())