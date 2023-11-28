import numpy as np

import cvxpy as cp
from cvxpy import Variable
from cvxpy.atoms.affine.kron import kron
from cvxpy.constraints import OpRelEntrConeQuad


def quantum_rel_entr_canon(expr, args):
    X, Y = args
    n = X.shape[0]
    Imat = np.eye(n)
    e = Imat.ravel().reshape(n ** 2, 1)
    if (not X.is_real()) or (not Y.is_real()):
        assert X.is_hermitian()
        assert Y.is_hermitian()
        first_arg = cp.atoms.affine.wraps.hermitian_wrap(kron(X, Imat))
        second_arg = cp.atoms.affine.wraps.hermitian_wrap(kron(Imat, Y.conj()))
        epi = Variable(shape=first_arg.shape, hermitian=True)
        # TODO
        #   1. appropriately canonicalize first_arg, second_arg, and epi
        #      into real and imaginary parts.
        #   2. call the op_rel_entr_cone_canon function from
        #       cvxpy/reductions/complex2real/canonicalizers/matrix_canon.py
        #   3. extract the equivalent real OpRelEntrConeQuad constraint.
        raise NotImplementedError('Finish me!')
    else:
        assert X.is_symmetric()
        assert Y.is_symmetric()
        first_arg = cp.atoms.affine.wraps.symmetric_wrap(kron(X, Imat))
        second_arg = cp.atoms.affine.wraps.symmetric_wrap(kron(Imat, Y))
        epi = Variable(shape=first_arg.shape, symmetric=True)
        orec_con = OpRelEntrConeQuad(first_arg, second_arg, epi,
                                     expr.quad_approx[0], expr.quad_approx[1]
        )
    # at this point we can be certain that we're dealing with an OpRelEnterConeQuad
    # constraint with real inputs. Canonicalize it, and return the results!
    main_con, aux_cons = cp.reductions.cone2cone.approximations.OpRelEntrConeQuad_canon(
        orec_con, None
    )
    constrs = [main_con] + aux_cons
    return -e.T @ epi @ e, constrs


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
