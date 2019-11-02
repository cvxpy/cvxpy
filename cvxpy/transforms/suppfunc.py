from cvxpy.expressions.variable import Variable
from cvxpy.atoms.suppfunc import SuppFuncAtom
import numpy as np
from scipy import sparse


def scs_coniclift(x, constraints):
    """
    Return (A, b, K) so that
        {x : x satisfies constraints}
    can be written as
        {x : exists y where A @ [x; y] + b in K}.

    Parameters
    ----------
    x: cvxpy.Variable
    constraints: list of cvxpy.constraints.constraint.Constraint
        Each Constraint object must be DCP-compatible.

    Notes
    -----
    This function DOES NOT work when ``x`` has attributes, like ``PSD=True``,
    ``diag=True``, ``symmetric=True``, etc...
    """
    from cvxpy.problems.problem import Problem
    from cvxpy.problems.objective import Minimize
    from cvxpy.atoms.affine.sum import sum
    prob = Problem(Minimize(sum(x)), constraints)
    # ^ The objective value is only used to make sure that "x"
    # participates in the problem. So, if constraints is an
    # empty list, then the support function is the standard
    # support function for R^n.
    data, chain, invdata = prob.get_problem_data(solver='SCS')
    inv = invdata[-2]
    x_offset = inv.var_offsets[x.id]
    x_indices = np.arange(x_offset, x_offset + x.size)
    A = data['A']
    x_selector = np.zeros(shape=(A.shape[1],), dtype=bool)
    x_selector[x_indices] = True
    A_x = A[:, x_selector]
    A_other = A[:, ~x_selector]
    A = -sparse.hstack([A_x, A_other])
    b = data['b']
    K = data['dims']
    return A, b, K


def scs_cone_selectors(K):
    """
    Parse a ConeDims object, as returned from SCS's apply function.

    Return a dictionary which gives row-wise information for the affine
    operator returned from SCS's apply function.

    Parameters
    ----------
    K : cvxpy.reductions.solvers.conic_solver.ConeDims

    Returns
    -------
    selectors : dict
        Keyed by strings, which specify cone types. Values are numpy
        arrays, or lists of numpy arrays. The numpy arrays give row indices
        of the affine operator (A, b) returned by SCS's apply function.
    """
    idx = K.zero
    nonpos_idxs = np.arange(idx, idx + K.nonpos)
    idx += K.nonpos
    # ^ Called "nonpos", but its actually nonneg.
    soc_idxs = []
    for soc in K.soc:
        idxs = np.arange(idx, idx + soc)
        soc_idxs.append(idxs)
        idx += soc
    psd_idxs = []
    for psd in K.psd:
        veclen = psd * (psd + 1) // 2
        psd_idxs.append(np.arange(idx, idx + veclen))
        idx += veclen
    expsize = 3 * K.exp
    exp_idxs = np.arange(idx, idx + expsize)
    selectors = {
        'nonneg': nonpos_idxs,
        'exp': exp_idxs,
        'soc': soc_idxs,
        'psd': psd_idxs
    }
    return selectors


class SuppFunc(object):

    def __init__(self, x, constraints):
        """
        A callable python object, representing the support function of the convex set:

            S = { val : it is possible to satisfy the given constraints, when x.value = val }.

        Parameters
        ----------
        x : cvxpy.Variable
            This variable cannot have any attributes, such as PSD=True, nonneg=True,
            symmetric=True, etc...

        constraints : list of cvxpy.constraints.constraint.Constraint
            Usually, these are constraints over ``x``, and some number of auxiliary
            cvxpy Variables. It is valid to supply ``constraints=[]``.
        """
        if len(constraints) == 0:
            dummy = Variable()
            constraints.append(dummy == 1)
        A, b, K = scs_coniclift(x, constraints)
        K_sels = scs_cone_selectors(K)
        self._A = A
        self._b = b
        self._K_sels = K_sels
        self._x = x
        self._constraints = constraints
        pass

    def __call__(self, y):
        """
        Return an atom representing

            max{ cvxpy.vec(y) @ cvxpy.vec(x) : x in S }

        where S is the convex set associated with this SuppFunc object.
        """
        sigma_at_y = SuppFuncAtom(y, self._A, self._b, self._K_sels,
                                  self._x, self._constraints)
        return sigma_at_y
