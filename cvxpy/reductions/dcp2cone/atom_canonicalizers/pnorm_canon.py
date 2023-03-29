"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from fractions import Fraction

import numpy as np

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import (
    abs_canon,
)
from cvxpy.utilities.power_tools import gm_constrs


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
