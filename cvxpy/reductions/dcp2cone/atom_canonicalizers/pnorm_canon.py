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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.power_tools import gm_constrs
from fractions import Fraction
import numpy as np


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.p
    axis = expr.axis
    shape = expr.shape
    t = Variable(shape)

    if p == 2:
        if axis is None:
            assert shape == tuple()
            return t, [SOC(t, vec(x))]
        else:
            return t, [SOC(vec(t), x, axis)]

    # we need an absolute value constraint for the symmetric convex branches
    # (p > 1)
    constraints = []
    if p > 1:
        # TODO(akshayka): Express this more naturally (recursively), in terms
        # of the other atoms
        abs_expr = abs(x)
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    # now, we take care of the remaining convex and concave branches
    # to create the rational powers, we need a new variable, r, and
    # the constraint sum(r) == t
    r = Variable(x.shape)
    constraints += [sum(r) == t]

    # todo: no need to run gm_constr to form the tree each time.
    # we only need to form the tree once
    promoted_t = Constant(np.ones(x.shape)) * t
    p = Fraction(p)
    if p < 0:
        constraints += gm_constrs(promoted_t, [x, r],  (-p/(1-p), 1/(1-p)))
    if 0 < p < 1:
        constraints += gm_constrs(r,  [x, promoted_t], (p, 1-p))
    if p > 1:
        constraints += gm_constrs(x,  [r, promoted_t], (1/p, 1-1/p))

    return t, constraints
