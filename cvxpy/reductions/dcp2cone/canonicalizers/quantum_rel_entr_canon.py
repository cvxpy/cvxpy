from cvxpy import Variable
from cvxpy.atoms.affine.kron import kron
from cvxpy.constraints import OpRelEntrConeQuad
from cvxpy.reductions.cone2cone.approximations import gauss_legendre

import cvxpy as cp
import numpy as np

def quantum_rel_entr_canon(expr, args):
    constrs = []
    X, Y = args
    n = X.shape[0]
    I = np.eye(n)
    e = I.ravel().reshape(n ** 2, 1)
    first_arg = cp.atoms.affine.wraps.symmetric_wrap(kron(X, I))
    second_arg = cp.atoms.affine.wraps.symmetric_wrap(kron(I, Y.conj()))
    epi = Variable(shape=first_arg.shape, symmetric=True)
    constrs.append(OpRelEntrConeQuad(first_arg, second_arg, epi,
                                     expr.quad_approx[0], expr.quad_approx[1]))
    return e.T @ epi @ e, constrs


### Attempt at implementing smaller canonicalization of P(r_m)
# def quantum_rel_entr_canon(expr, args):
#     constrs = []
#     m, _ = expr.quad_approx
#     X, Y = args
#     n = X.shape[0]
#     I = np.eye(n)
#     e = I.ravel().reshape(n ** 2, 1)
#     first_arg = kron(X, I)
#     second_arg = kron(I, Y.conj())
#     taus = {i: Variable() for i in range(m+1)}
#     w, t = gauss_legendre(m)

#     constrs.append(cp.Zero(cp.sum([w[i] * taus[i] for i in range(m)]) - taus[m]))
#     for i in range(m):
#         block_11 = second_arg + t[i] * (first_arg - second_arg)
#         block_12 = second_arg @ e
#         block_21 = e.T @ second_arg
#         block_22 = e.T @ second_arg @ e - t[i] * taus[i]
#         constrs.append(cp.bmat([[block_11, block_12], [block_21, block_22]]) >> 0)

#     # negative sign was an experiment, get rid of it to get the correct hypograph
#     # of phi(perspective(r_m))
#     return -taus[m], constrs




### (feasible) test problem!
# import numpy as np
# import cvxpy as cp

# np.random.seed(0)
# X = cp.Variable(shape=(3,3), symmetric=True)
# A = np.random.randn(3,3)

# obj = cp.Minimize(cp.quantum_rel_entr(X, np.eye(3)))
# cons = [
#     X >> 0,
#     cp.norm(X @ A) <= 10
# ]

# prob = cp.Problem(obj, cons)
# prob.solve(solver='MOSEK')
