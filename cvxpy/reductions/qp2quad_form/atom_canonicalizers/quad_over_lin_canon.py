from cvxpy.expressions.variables import Variable
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.quad_form import SymbolicQuadForm
from numpy import eye

def quad_over_lin_canon(expr, args):
    x = args[0]
    y = args[1]
    shape = x.shape
    size = shape[0]*shape[1]
    t = Variable(size, 1)
    return SymbolicQuadForm(t, expr), [reshape(t, shape) == x]