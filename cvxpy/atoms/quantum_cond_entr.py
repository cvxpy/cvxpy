from typing import Optional

import numpy as np

from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.atoms.affine.wraps import hermitian_wrap
from cvxpy.atoms.quantum_rel_entr import quantum_rel_entr
from cvxpy.expressions.expression import Expression


def quantum_cond_entr(rho: Expression , dim: list[int], sys: Optional[int]=0):
    if sys == 0:
        composite_arg = kron(np.eye(dim[0]),
                             partial_trace(rho, dim, sys))
        return -quantum_rel_entr(rho, hermitian_wrap(composite_arg))
    elif sys == 1:
        composite_arg = kron(partial_trace(rho, dim, sys),
                             np.eye(dim[1]))
        return -quantum_rel_entr(rho, hermitian_wrap(composite_arg))
