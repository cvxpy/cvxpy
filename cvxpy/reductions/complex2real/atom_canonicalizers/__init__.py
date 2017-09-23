"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms import (bmat, cumsum, diag, kron, conv,
                         promote, abs, reshape, trace,
                         neg, upper_tri, conj, imag, real,
                         norm1, norm_inf, norm2, Pnorm)
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms.affine.binary_operators import (MulExpression,
                                                 multiply,
                                                 DivExpression)
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from abs_canon import abs_canon
from aff_canon import (separable_canon, real_canon,
                       imag_canon, conj_canon, binary_canon)
from pnorm_canon import pnorm_canon
from variable_canon import variable_canon
from constant_canon import constant_canon


CANON_METHODS = {
    AddExpression: separable_canon,
    bmat: separable_canon,
    cumsum: separable_canon,
    diag: separable_canon,
    Hstack: separable_canon,
    index: separable_canon,
    promote: separable_canon,
    reshape: separable_canon,
    Sum: separable_canon,
    trace: separable_canon,
    transpose: separable_canon,
    neg: separable_canon,
    upper_tri: separable_canon,
    Vstack: separable_canon,

    conv: binary_canon,
    DivExpression: binary_canon,
    kron: binary_canon,
    MulExpression: binary_canon,
    multiply: binary_canon,

    conj: conj_canon,
    imag: imag_canon,
    real: real_canon,
    Variable: variable_canon,
    Constant: constant_canon,

    abs: abs_canon,
    norm1: pnorm_canon,
    norm_inf: pnorm_canon,
    norm2: pnorm_canon,
    Pnorm: pnorm_canon,
}
