from cvxpy.atoms.affine.trace import trace
#from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variables.variable import Variable


def matrix_frac_canon(expr, args):
    X = args[0]  # n by m matrix.
    P = args[1]  # n by n matrix.
    n, m = X.shape
    # Create a matrix with Schur complement T - X.T*P^-1*X.
    M = Variable(n+m, n+m)
    T = Variable(m, m)
    constraints = []
    # Fix M using the fact that P must be affine by the DCP rules.
    # M[0:n, 0:n] == P.
    constraints.append(M[0:n, 0:n] == P)
    # M[0:n, n:n+m] == X
    constraints.append(M[0:n, n:n+m] == X)
    # M[n:n+m, n:n+m] == T
    constraints.append(M[n:n+m, n:n+m] == T)
    # Add SDP constraint.
    constraints.append(PSD(M))
    return trace(T), constraints
