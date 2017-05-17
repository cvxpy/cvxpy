from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variables import Variable

import numpy as np

def power_canon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p != 2:
        raise ValueError("quadratic form can only have power 2")
    t = Variable(x.size, 1)
    return SymbolicQuadForm(t, expr), [reshape(t, x.shape) == x]