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

from cvxpy.atoms import *
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from abs_canon import abs_canon
from aff_canon import (separable_canon, real_canon,
                       imag_canon, conj_canon)
from norm1_canon import norm1_canon
from norm_inf_canon import norm_inf_canon
from variable_canon import variable_canon


CANON_METHODS = {
    add_expr : separable_canon,
    bmat : separable_canon,
    cumsum : separable_canon,
    diag : separable_canon,
    hstack : separable_canon,
    index : separable_canon,
    kron : separable_canon,
    promote : separable_canon,
    reshape : separable_canon,
    sum : separable_canon,
    trace : separable_canon,
    transpose : separable_canon,
    neg : separable_canon,
    upper_tri : separable_canon,
    vec : separable_canon,
    vstack : separable_canon,

    conv : TODO,
    DivExpression : TODO,
    MulExpression : TODO,
    multiply : TODO,

    conj : conj_canon,
    imag : imag_canon,
    real : real_canon,
    Variable : variable_canon,
    Constant : constant_canon,

    abs : abs_canon,
    norm1 : norm1_canon,
    norm_inf : norm_inf_canon,
}
