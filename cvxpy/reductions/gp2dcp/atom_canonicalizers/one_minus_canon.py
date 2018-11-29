from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.exp import exp


def one_minus_canon(expr, args):
    del expr
    return log(1 - exp(args[0])), []
