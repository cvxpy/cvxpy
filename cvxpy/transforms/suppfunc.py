from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from cvxpy.atoms.suppfunc import SuppFuncAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD, SvecPSD
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CONVEX_ATTRIBUTES
from cvxpy.utilities.solver_context import SolverInfo

if TYPE_CHECKING:
    from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims


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


def _coniclift(
    x: Variable,
    constraints: list[Constraint],
    solver_context: SolverInfo,
) -> tuple[
    sparse.sparray,
    np.ndarray,
    dict[str, np.ndarray | list],
]:
    """
    Return (A, b, cone selectors) so that
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
    from cvxpy.constraints.finite_set import FiniteSet
    from cvxpy.error import SolverError
    from cvxpy.problems.objective import Minimize
    from cvxpy.problems.problem import Problem
    from cvxpy.problems.problem_form import ProblemForm
    from cvxpy.reductions.chain import Chain
    from cvxpy.reductions.complex2real import complex2real
    from cvxpy.reductions.cone2cone.approx import ApproxCone2Cone
    from cvxpy.reductions.cone2cone.exact import ExactCone2Cone
    from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
    from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import (
        ConeMatrixStuffing,
    )
    from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
    from cvxpy.reductions.eliminate_zero_sized import EliminateZeroSized
    from cvxpy.reductions.solvers.solver import expand_cones

    prob = Problem(Minimize(sum(x)), constraints)
    has_finite_set = any(isinstance(con, FiniteSet) for con in constraints)
    if prob.is_mixed_integer() or has_finite_set:
        raise SolverError(
            "SuppFunc does not support mixed-integer set descriptions.")
    # ^ The objective value is only used to make sure that "x"
    # participates in the problem. So, if constraints is an
    # empty list, then the support function is the standard
    # support function for R^n.
    problem_form = ProblemForm(prob)
    cones = problem_form.cones(quad_obj=False).copy()
    _, exact_targets, approx_targets = expand_cones(
        cones, solver_context.solver_supported_constraints)

    # This deliberately mirrors the parameter-free DCP portion of the normal
    # solving chain, but stops before solver-specific cone formatting. Bounds
    # must be explicit because they are part of the set being dualized.
    reductions = []
    if complex2real.accepts(prob):
        reductions.append(complex2real.Complex2Real())
    reductions.extend([
        Dcp2Cone(quad_obj=False, solver_context=solver_context),
        CvxAttr2Constr(reduce_bounds=True),
    ])
    if exact_targets:
        reductions.append(ExactCone2Cone(
            target_cones=exact_targets, solver_context=solver_context))
    if approx_targets:
        reductions.append(ApproxCone2Cone(target_cones=approx_targets))
    reductions.extend([
        EliminateZeroSized(),
        ConeMatrixStuffing(quad_obj=False),
    ])

    cone_prog, _ = Chain(reductions=reductions).apply(prob)
    _, _, A, b = cone_prog.apply_parameters()
    x_offset = cone_prog.var_id_to_col[x.id]
    x_indices = np.arange(x_offset, x_offset + x.size)
    x_selector = np.zeros(shape=(A.shape[1],), dtype=bool)
    x_selector[x_indices] = True
    A_x = A[:, x_selector]
    A_other = A[:, ~x_selector]
    A = sparse.hstack([A_x, A_other])
    return A, b, _cone_selectors(cone_prog.cone_dims, cone_prog.constraints)


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


def _cone_selectors(
    K: "ConeDims", constraints: list[Constraint],
) -> dict[str, np.ndarray | list]:
    """
    Parse a ConeDims object from an unformatted ParamConeProg.

    Return a dictionary which gives row-wise information for the affine
    operator stored by the ParamConeProg.

    Parameters
    ----------
    K : cvxpy.reductions.dcp2cone.cone_matrix_stuffing.ConeDims
    constraints : list[Constraint]
        Ordered constraints from the unformatted ParamConeProg.

    Returns
    -------
    selectors : dict
        Keyed by strings, which specify cone types. Values are numpy
        arrays, or lists of numpy arrays. The numpy arrays give row indices
        of the affine operator (A, b) stored by the ParamConeProg.
    """
    if K.p3d or K.pnd:
        msg = "SuppFunc doesn't yet support feasible sets represented \n"
        msg += "with power cone constraints."
        raise NotImplementedError(msg)
        # TODO: implement
    idx = K.zero
    nonneg_idxs = np.arange(idx, idx + K.nonneg)
    idx += K.nonneg
    soc_idxs = []
    psd_idxs = []
    exp_idxs = []
    for con in constraints:
        match con:
            case SOC():
                cone_count = con.num_cones()
                rows = np.arange(idx, idx + con.size)
                soc_idxs.extend(np.column_stack((
                    rows[:cone_count],
                    rows[cone_count:].reshape(cone_count, -1),
                )))
                idx += con.size
            case PSD() | SvecPSD():
                psd_idxs.append((np.arange(idx, idx + con.size), con))
                idx += con.size
            case ExpCone():
                cone_count = con.num_cones()
                exp_idxs.extend(
                    np.arange(idx, idx + con.size).reshape(3, cone_count).T.ravel()
                )
                idx += con.size
    selectors = {
        'nonneg': nonneg_idxs,
        'exp': np.asarray(exp_idxs, dtype=int),
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
        self.constraints = list(constraints)
        self._solver_context = None
        self._conic_repr = None
        self._scs_conic_repr = None

    def __call__(self, y) -> SuppFuncAtom:
        """
        Return an atom representing

            max{ cvxpy.vec(y) @ cvxpy.vec(x) : x in S }

        where S is the convex set associated with this SuppFunc object.
        """
        sigma_at_y = SuppFuncAtom(y, self)
        return sigma_at_y

    def _conic_repr_of_set(
        self, solver_context: SolverInfo,
    ) -> tuple[
        sparse.sparray,
        np.ndarray,
        dict[str, np.ndarray | list],
    ]:
        # Proper cross-chain caching is deferred for now; this only caches
        # canonicalizations that share the same SolverInfo instance.
        if self._solver_context is solver_context:
            return self._conic_repr
        if len(self.constraints) == 0:
            dummy = Variable()
            constrs = [dummy == 1]
        else:
            constrs = self.constraints
        conic_repr = _coniclift(self.x, constrs, solver_context)
        self._conic_repr = conic_repr
        self._solver_context = solver_context
        return conic_repr

    def conic_repr_of_set(self):
        """Return the historical SCS-formatted representation of the set."""
        if self._scs_conic_repr is None:
            if len(self.constraints) == 0:
                dummy = Variable()
                constrs = [dummy == 1]
            else:
                constrs = self.constraints
            A, b, K = scs_coniclift(self.x, constrs)
            self._scs_conic_repr = (A, b, scs_cone_selectors(K))
        return self._scs_conic_repr
