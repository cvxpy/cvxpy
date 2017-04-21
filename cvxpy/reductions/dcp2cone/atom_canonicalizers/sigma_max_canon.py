#from cvxpy.constraints.semidefinite import SDP
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def sigma_max_canon(expr, args):
    A = arg[0]
    n, m = A.shape
    X = Variable(n+m, n+m)

    shape = expr.shape
    t = Variable(*shape)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:n, 0:n] == I_n*t
    constraints.append(X[0:n, 0:n] == np.eye(n) * t)

    # X[0:n, n:n+m] == A
    constraints.append(X[0:n, n:n+m] == A)

    # X[n:n+m, n:n+m] == I_m*t
    constraints.append(X[n:n+m, n:n+m] == np.eye(m) * t)

    # SDP constraint
    constraints.append(PSD(X))

    return t, constraints
