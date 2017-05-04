from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.utilities.power_tools import gm_constrs
from fractions import Fraction
import numpy as np


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.p
    axis = expr.axis
    shape = expr.shape
    t = Variable(*shape)

    if p == 2:
        if axis is None:
            assert shape == (1, 1)
            return t, [SOC(t, x)]
        else:
            return t, [SOC(reshape(t, (shape[0] * shape[1], 1)), x, axis)]
    if p == np.inf:
        promoted_t = Constant(np.ones(x.shape)) * t
        return t, [x <= promoted_t, x + promoted_t >= 0]

    # we need an absolute value constraint for the symmetric convex branches 
    # (p >= 1)
    constraints = []
    if p >= 1:
        # TODO(akshayka): Express this more naturally (recursively), in terms
        # of the other atoms
        abs_expr = abs(Variable(*x.shape))
        abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
        x = abs_x
        constraints += abs_constraints

    if p == 1:
        return sum_entries(x), constraints

    # now, we take care of the remaining convex and concave branches
    # to create the rational powers, we need a new variable, r, and
    # the constraint sum(r) == t
    r = Variable(*x.shape)
    constraints += [sum_entries(r) == t]

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
