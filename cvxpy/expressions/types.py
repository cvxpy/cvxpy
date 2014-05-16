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

# Utility functions to solve circular imports.
def constant():
    import constants
    return constants.Constant

def add_expr():
    from ..atoms.affine import add_expr
    return add_expr.AddExpression

def mul_expr():
    from ..atoms.affine import binary_operators
    return binary_operators.MulExpression

def div_expr():
    from ..atoms.affine import binary_operators
    return binary_operators.DivExpression

def neg_expr():
    from ..atoms.affine import unary_operators
    return unary_operators.NegExpression

def index():
    from ..atoms.affine import index
    return index.index

def transpose():
    from ..atoms.affine import transpose
    return transpose.transpose
