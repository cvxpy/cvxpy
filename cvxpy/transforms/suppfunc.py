import numpy as np
from scipy import sparse

from cvxpy.atoms.suppfunc import SuppFuncAtom
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CONVEX_ATTRIBUTES


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
    from cvxpy.atoms.affine.sum import sum
    from cvxpy.problems.objective import Minimize
    from cvxpy.problems.problem import Problem
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
    if K.p3d:
        msg = "SuppFunc doesn't yet support feasible sets represented \n"
        msg += "with power cone constraints."
        raise NotImplementedError(msg)
        # TODO: implement
    idx = K.zero
    nonneg_idxs = np.arange(idx, idx + K.nonneg)
    idx += K.nonneg
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
        'nonneg': nonneg_idxs,
        'exp': exp_idxs,
        'soc': soc_idxs,
        'psd': psd_idxs
    }
    return selectors


class SuppFunc:
    """
    Given a list of CVXPY Constraint objects :math:`\\texttt{constraints}`
    involving a real CVXPY Variable :math:`\\texttt{x}`, consider the convex set

    .. math::

        S = \\{ v : \\text{it's possible to satisfy all } \\texttt{constraints}
                    \\text{ when } \\texttt{x.value} = v \\}.

    This object represents the *support function* of :math:`S`.
    This is the convex function

    .. math::

        y \\mapsto \\max\\{ \\langle y, v \\rangle : v \\in S \\}.

    The support function is a fundamental object in convex analysis.
    It's extremely useful for expressing dual problems using
    `Fenchel duality <https://en.wikipedia.org/wiki/Fenchel%27s_duality_theorem>`_.

    Parameters
    ----------
    x : Variable
        This variable cannot have any attributes, such as PSD=True, nonneg=True,
        symmetric=True, etc...

    constraints : list[Constraint]
        Usually, these are constraints over :math:`\\texttt{x}`, and some number of auxiliary
        CVXPY Variables. It is valid to supply :math:`\\texttt{constraints = []}`.

    Examples
    --------
    If :math:`\\texttt{h = cp.SuppFunc(x, constraints)}`, then you can use
    :math:`\\texttt{h}` just like any other scalar-valued atom in CVXPY.
    For example, if :math:`\\texttt{x}` was a CVXPY Variable with
    :math:`\\texttt{x.ndim == 1}`, you could do the following:

    .. code::

        z = cp.Variable(shape=(10,))
        A = np.random.standard_normal((x.size, 10))
        c = np.random.rand(10)
        objective =  h(A @ z) - c @ z
        prob = cp.Problem(cp.Minimize(objective), [])
        prob.solve()

    Notes
    -----
    You are allowed to use CVXPY Variables other than :math:`\\texttt{x}` to define
    :math:`\\texttt{constraints}`, but the set :math:`S` only consists of objects
    (vectors or matrices) with the same shape as :math:`\\texttt{x}`.

    It's possible for the support function to take the value :math:`+\\infty`
    for a fixed vector :math:`\\texttt{y}`. This is an important point, and
    it's one reason why support functions are actually formally defined with
    the supremum ":math:`\\sup`" rather than the maximum ":math:`\\max`".
    For more information on support functions, check out
    `this Wikipedia page <https://en.wikipedia.org/wiki/Support_function>`_.
    """

    def __init__(self, x, constraints):
        if not isinstance(x, Variable):
            raise ValueError('The first argument must be an unmodified cvxpy Variable object.')
        if any(x.attributes[attr] for attr in CONVEX_ATTRIBUTES):
            raise ValueError('The first argument cannot have any declared attributes.')
        for con in constraints:
            con_params = con.parameters()
            if len(con_params) > 0:
                raise ValueError('Convex sets described with Parameter objects are not allowed.')
        self.x = x
        self.constraints = constraints
        self._A = None
        self._b = None
        self._K_sels = None
        self._compute_conic_repr_of_set()

    def __call__(self, y) -> SuppFuncAtom:
        """
        Return an atom representing

            max{ cvxpy.vec(y) @ cvxpy.vec(x) : x in S }

        where S is the convex set associated with this SuppFunc object.
        """
        sigma_at_y = SuppFuncAtom(y, self)
        return sigma_at_y

    def _compute_conic_repr_of_set(self) -> None:
        if len(self.constraints) == 0:
            dummy = Variable()
            constrs = [dummy == 1]
        else:
            constrs = self.constraints
        A, b, K = scs_coniclift(self.x, constrs)
        K_sels = scs_cone_selectors(K)
        self._A = A
        self._b = b
        self._K_sels = K_sels

    def conic_repr_of_set(self):
        return self._A, self._b, self._K_sels
