from cvxpy import Variable
from cvxpy.atoms.affine.kron import kron
from cvxpy.constraints import OpRelEntrConeQuad
import numpy as np

def quantum_rel_entr_canon(expr, args):
    constrs = []
    X, Y = args
    n = X.shape[0]
    I = np.eye(n)
    first_arg = kron(X, I)
    second_arg = kron(I, Y.conj())
    epi = Variable(shape=first_arg.shape)
    constrs.append(OpRelEntrConeQuad(first_arg, second_arg, epi,
                                     expr.quad_approx[0], expr.quad_approx[1]))
    return epi, constrs
