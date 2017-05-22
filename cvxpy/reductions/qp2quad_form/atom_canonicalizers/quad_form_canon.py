from cvxpy.atoms import reshape
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variables import Variable

def quad_form_canon(expr, args):
    x = expr.args[0]
    P = expr.args[1]
    shape = x.shape
    size = x.shape[0]*x.shape[1]
    t = Variable(size, 1)
    return SymbolicQuadForm(t, P, expr), [reshape(t, shape) == x]
