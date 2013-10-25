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
from ..constraints.affine import AffEqConstraint
import operator as op

class AffExpression(u.Affine):
    """ An affine expression. The result of canonicalization. """
    # coefficients - a dict of {variable/Constant: [parameter expressions]}
    # shape - the shape of the expresssion.
    def __init__(self, coefficients, shape):
        self._coeffs = coefficients
        self._shape = shape
        super(AffExpression, self).__init__()

    def __str__(self):
        return str(self._coeffs)

    def __repr__(self):
        return str(self)

    # The dimensions of the expression.
    @property
    def size(self):
        return self._shape.size

    # Returns a list of the variables in the expression.
    def variables(self):
        variables = self._coeffs.keys()
        if s.CONSTANT in variables:
            variables.remove(s.CONSTANT)
        return variables

    # Returns a coefficient dict with all parameter expressions evaluated.
    def coefficients(self):
        new_coeffs = {}
        for var,blocks in self._coeffs.items():
            new_coeffs[var] = map(self.eval_expr, blocks)
        return new_coeffs

    # Helper function to evaluate parameter expressions.
    @staticmethod
    def eval_expr(expr):
        try:
            return expr.value
        except Exception, e:
            return expr

    # # Returns a dict of term id to coefficient.
    # # interface - the matrix interface to convert constants
    # #             into a matrix of the target class.
    # def coefficients(self, interface):
    #     final_coeffs = {}
    #     for mults in self._terms:
    #         root = mults[0]
    #         root_coeffs = root.coefficients(interface)
    #         for id,coeff in root_coeffs.items():
    #             coefficient = self.dequeue_mults(coeff, mults, interface)
    #             if id in final_coeffs:
    #                 final_coeffs[id] = final_coeffs[id] + coefficient
    #             else:
    #                 final_coeffs[id] = coefficient
    #     return final_coeffs

    # # Resolves a multiplication stack into a single coefficient.
    # @staticmethod
    # def dequeue_mults(coefficient, mults, interface):
    #     for i in range(len(mults)-1):
    #         lh = mults[i+1]
    #         # Only contains a constant coefficient.
    #         lh_coeff = lh.coefficients(interface).values()[0]
    #         coefficient = lh_coeff * coefficient
    #     return coefficient

    # Multiplies by a ones matrix to promote a scalar coefficient.
    # Returns an updated AffExpression.
    @staticmethod
    def promote(coeffs, shape):
        rows,cols = shape.size
        ones = intf.DEFAULT_NP_INTERFACE.ones(rows)
        new_coeffs = {}
        for var,blocks in coeffs.items():
            new_coeffs[var] = [ones*blocks[0] for i in range(cols)]
        return AffExpression(new_coeffs, shape)

    # Combines the dicts. Adds the blocks of common variables.
    # Multiplies all blocks by a ones vector if promotion occurs.
    def __add__(self, other):
        new_shape = self._shape + other._shape
        if new_shape.size > self._shape.size:
            self = self.promote(self, new_shape)
        elif new_shape.size > other._shape.size:
            other = self.promote(other, new_shape)
        # Merge the dicts, summing common variables.
        new_coeffs = self._coeffs.copy()
        for var,blocks in other._coeffs.items():
            if var in new_coeffs:
                block_sum = []
                for (b1,b2) in zip(self._coeffs[var], other._coeffs[var]):
                    block_sum.append(b1 + b2)
                new_coeffs[var] = block_sum
            else:
                new_coeffs[var] = blocks
        return AffExpression(new_coeffs, new_shape)

    def __sub__(self, other):
        return self + -other

    # # Distributes multiplications by left hand terms
    # # across right hand terms.
    # def __mul__(self, other):
    #     terms = AffExpression.mul_terms(self._terms, other._terms)
    #     return AffExpression(other.variables(), terms, 
    #                         self._shape * other._shape)

    # Negates every parameter expression.
    def __neg__(self):
        new_coeffs = {}
        for var,blocks in self._coeffs.items():
            new_coeffs[var] = map(op.neg, blocks)
        return AffExpression(new_coeffs, self._shape)

    # # Utility function for multiplying lists of terms.
    # @staticmethod
    # def mul_terms(lh_terms, rh_terms):
    #     terms = []
    #     for lh_mult in lh_terms:
    #         for rh_mult in rh_terms:
    #             mult = deque(rh_mult)
    #             mult.extend(lh_mult)
    #             terms.append(mult)
    #     return terms


    # # Returns an (AffineObjective, [AffineConstraints]) tuple
    # # representing the tranpose.
    # @property
    # def T(self):
    #     A = types.variable()(*self.size)
    #     obj = A.T.canonical_form()[0]
    #     return (obj, [AffEqConstraint(A, self)])