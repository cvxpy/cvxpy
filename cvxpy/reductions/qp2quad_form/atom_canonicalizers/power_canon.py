from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.expressions.variables import Variable

import numpy as np

def power_canon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p != 2:
        raise ValueError("quadratic form can only have power 2")
    expr = []
    con = []
    one = np.array([[1.0]])
    t = Variable(x.size)
    return vstack(*[QuadForm(t[i], one) for i in range(x.size)]), [reshape(t, x.shape) == x]