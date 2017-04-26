from cvxpy.atoms.affine.trace import trace
#from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variables.variable import Variable


def normNuc_canon(expr, args):
    A = args[0]
    m, n = A.shape

    # Create the equivalent problem:
    #   minimize (trace(U) + trace(V))/2
    #   subject to:
    #            [U A; A.T V] is positive semidefinite
    X = Variable(m+n, m+n)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:rows,rows:rows+cols] == A
    constraints.append(X[0:m, m:m+n] == A)
    constraints.append(PSD(X))
    trace_value = 0.5 * trace(X)

    return trace_value, constraints
