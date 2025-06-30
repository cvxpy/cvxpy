from typing import Literal, Tuple

import numpy as np

from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.atoms.affine.wraps import hermitian_wrap
from cvxpy.atoms.quantum_rel_entr import quantum_rel_entr
from cvxpy.expressions.expression import Expression


def quantum_cond_entr(
        rho: Expression , dims: Tuple[int, int], sys: Literal[0, 1] = 0, quad_approx=(3, 3)
    ):
    """
    Returns (an approximation of) the quantum conditional entropy for a bipartite state,
    conditioning on system :math:`\\texttt{sys}.`

    Formally, if :math:`N` is the von Neumann entropy function and
    :math:`\\operatorname{tr}_{\\texttt{sys}}` is the partial trace operator over subsystem
    :math:`\\texttt{sys},` the returned expression represents

    .. math::
        N(\\rho) - N(\\operatorname{tr}_{\\texttt{sys}}(\\rho)).

    Parameters
    ----------
    rho : Expression
        A Hermitian matrix of order :math:`\\texttt{dims[0]}\\cdot\\texttt{dims[1]}.`

    dims : tuple
        The dimensions of the two subsystems that definte :math:`\\rho` as a bipartite state.

    sys : int
        The subsystem on which to condition in evaluating the conditional quantum entropy.

    quad_approx : Tuple[int, int]
        quad_approx[0] is the number of quadrature nodes and quad_approx[1] is the number of scaling
        points in the quadrature scheme from https://arxiv.org/abs/1705.00812.

    Notes
    -----
    This function does not assume :math:`\\operatorname{tr}(\rho)=1,` which would be required
    for most uses of this function in the context of quantum information theory. See 
    https://en.wikipedia.org/wiki/Conditional_quantum_entropy for more information. 
    """
    if len(dims) != 2:
        err = 'This function is only defined for a tensor product of two subsystems,' + \
        f'but {len(dims)} subsystems were implied from the value of dims.'
        raise ValueError(err)
    if sys == 0:
        composite_arg = kron(
            np.eye(dims[0]), partial_trace(rho, dims, sys)
        )
    elif sys == 1:
        composite_arg = kron(
            partial_trace(rho, dims, sys), np.eye(dims[1])
        )
    else:
        raise ValueError(f'Argument sys must be either 0 or 1; got {sys}.')
    composite_arg = hermitian_wrap(composite_arg)
    return -quantum_rel_entr(rho, composite_arg, quad_approx)
