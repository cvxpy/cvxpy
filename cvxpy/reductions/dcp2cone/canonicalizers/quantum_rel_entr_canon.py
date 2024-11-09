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
    # assert X.is_symmetric()
    # assert Y.is_symmetric()
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
    return  e.T @ epi @ e, constrs
