from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import (
    SOC,
    FiniteSet,
)
from cvxpy.error import DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem_form import ProblemForm, make_problem_form, pick_default_solver

if TYPE_CHECKING:
    from cvxpy.problems.problem import Problem
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approx import ApproxCone2Cone
from cvxpy.reductions.cone2cone.exact import ExactCone2Cone
from cvxpy.reductions.cone2cone.soc_dim3 import SOCDim3
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import (
    Valinvec2mixedint,
)
from cvxpy.reductions.eliminate_zero_sized import EliminateZeroSized
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver, expand_cones
from cvxpy.settings import COO_CANON_BACKEND, DPP_PARAM_THRESHOLD
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.warn import warn

DPP_ERROR_MSG = (
    "You are solving a parameterized problem that is not DPP. "
    "Because the problem is not DPP, subsequent solves will not be "
    "faster than the first one. For more information, see the "
    "documentation on Disciplined Parametrized Programming, at "
    "https://www.cvxpy.org/tutorial/dpp/index.html"
)


def _lookup_by_name(solver_name: str, gp: bool):
    """Look up installed QP and conic instances for a solver name.

    Returns (qp_inst | None, conic_inst | None). GP mode suppresses QP.
    """
    upper = solver_name.upper()
    qp = slv_def.SOLVER_MAP_QP.get(upper)
    if qp is not None and not qp.is_installed():
        qp = None
    conic = slv_def.SOLVER_MAP_CONIC.get(upper)
    if conic is not None and not conic.is_installed():
        conic = None
    if gp:
        qp = None
    return qp, conic


def _fallback_solver(problem_form: ProblemForm) -> Solver:
    """Last-resort: try every installed solver, warn, or raise."""
    for name in slv_def.INSTALLED_SOLVERS:
        if name in slv_def.COMMERCIAL_SOLVERS:
            continue
        for inst in (slv_def.SOLVER_MAP_CONIC.get(name),
                     slv_def.SOLVER_MAP_QP.get(name)):
            if inst is not None and inst.can_solve(problem_form):
                warn(
                    "The default solvers are not available. "
                    "Using %s. Consider specifying a solver "
                    "explicitly via the ``solver`` argument." %
                    inst.name()
                )
                return inst
    if problem_form.is_mixed_integer() and not problem_form.is_lp():
        warn(
            "Your problem is mixed-integer but not an LP. "
            "If your problem is nonlinear, consider installing SCIP "
            "(pip install pyscipopt) to solve it."
        )
    if problem_form.is_mixed_integer():
        raise SolverError(
            "You need a mixed-integer solver for this model. "
            "Refer to the documentation "
            "https://www.cvxpy.org/tutorial/advanced/"
            "index.html#mixed-integer-programs "
            "for discussion on this topic. "
            "Install the SCIP solver (pip install pyscipopt) "
            "for mixed-integer nonlinear problems, or HiGHS "
            "(pip install highspy) for mixed-integer LPs."
        )
    raise SolverError(
        "No installed solver could handle this problem."
    )


def _build_solving_chain(
    problem: Problem,
    solver_instance: Solver,
    problem_form: ProblemForm | None = None,
    gp: bool = False,
    enforce_dpp: bool = False,
    ignore_dpp: bool = False,
    canon_backend: str | None = None,
    solver_opts: dict | None = None,
) -> "SolvingChain":
    """Build a reduction chain for a specific solver.

    The chain is assembled in three stages:

    1. **Solver context** — ``SolverInfo`` is derived from
       *solver_instance* and *problem_form* (MIP-aware constraint set).
    2. **Pre-canonicalization reductions** — rewrites that depend only on
       *problem* and *gp* (complex→real, DGP→DCP, flip-objective,
       finite-set, DPP / EvalParams).
    3. **Canonicalization reductions** — determined by *problem_form*
       (required cones, quadratic objective) and *solver_context*
       (supported constraints, bounds, SOC-dim-3).

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    solver_instance : Solver
        The solver to target.
    problem_form : ProblemForm, optional
        Pre-computed structural analysis. Created automatically if not
        supplied.
    gp : bool
        If True, parse as a Disciplined Geometric Program.
    enforce_dpp : bool
        When True, raise DPPError for non-DPP problems.
    ignore_dpp : bool
        When True, treat DPP problems as non-DPP.
    canon_backend : str, optional
        Canonicalization backend ('CPP', 'SCIPY', or 'COO').
    solver_opts : dict, optional
        Solver-specific options.

    Returns
    -------
    SolvingChain
        A SolvingChain targeting the given solver.
    """
    if len(problem.variables()) == 0:
        return SolvingChain(reductions=[ConstantSolver()])

    if problem_form is None:
        problem_form = ProblemForm(problem)

    # Build SolverInfo up front — the canonicalization reductions below
    # are a function of (problem_form, DPP approach, solver_context).
    if problem_form.is_mixed_integer():
        supported = frozenset(
            getattr(solver_instance, 'MI_SUPPORTED_CONSTRAINTS',
                    solver_instance.SUPPORTED_CONSTRAINTS)
        )
    else:
        supported = frozenset(solver_instance.SUPPORTED_CONSTRAINTS)

    solver_context = SolverInfo(
        solver=solver_instance.name(),
        supported_constraints=supported,
        supports_bounds=solver_instance.BOUNDED_VARIABLES,
    )

    # --- Pre-canonicalization reductions (problem + gp only) ---
    reductions = []
    if complex2real.accepts(problem):
        reductions.append(complex2real.Complex2Real())
    if gp:
        reductions.append(Dgp2Dcp())
    if type(problem.objective) == Maximize:
        reductions.append(FlipObjective())
    constr_types = {type(c) for c in problem.constraints}
    if FiniteSet in constr_types:
        reductions.append(Valinvec2mixedint())

    # --- DPP handling ---
    dpp_context = 'dcp' if not gp else 'dgp'
    if ignore_dpp or not problem.is_dpp(dpp_context):
        if not ignore_dpp and enforce_dpp:
            raise DPPError(DPP_ERROR_MSG)
        if not ignore_dpp:
            warn(DPP_ERROR_MSG)
        reductions = [EvalParams()] + reductions
    else:
        if canon_backend is None:
            total_param_size = sum(p.size for p in problem.parameters())
            if total_param_size >= DPP_PARAM_THRESHOLD:
                canon_backend = COO_CANON_BACKEND

    # --- Canonicalization reductions (problem_form + solver_context) ---
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)

    # QP solvers always need quad_obj=True in the matrix stuffing step
    # because their apply() expects the P matrix from ConeMatrixStuffing.
    is_qp_solver = isinstance(solver_instance, QpSolver)
    quad_obj = (use_quad and solver_instance.supports_quad_obj()
                and (is_qp_solver or problem_form.has_quadratic_objective()))
    cones = problem_form.cones(quad_obj=quad_obj).copy()
    cones, exact_targets, approx_targets = expand_cones(cones, supported)

    reductions.append(Dcp2Cone(quad_obj=quad_obj, solver_context=solver_context))

    if exact_targets:
        reductions.append(ExactCone2Cone(target_cones=exact_targets))
    if approx_targets:
        reductions.append(ApproxCone2Cone(target_cones=approx_targets))

    reductions.append(
        CvxAttr2Constr(reduce_bounds=not solver_instance.BOUNDED_VARIABLES))
    reductions.append(EliminateZeroSized())

    if solver_instance.SOC_DIM3_ONLY and SOC in cones:
        reductions.append(SOCDim3())

    reductions += [
        ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend),
        solver_instance,
    ]
    return SolvingChain(reductions=reductions, solver_context=solver_context)


def _resolve_solver(
    solver,
    problem_form: ProblemForm,
    gp: bool,
) -> Solver:
    """Validate and resolve a solver argument to a concrete Solver instance.

    Parameters
    ----------
    solver : str, Solver, or None
        The solver to resolve.  ``None`` selects a default based on
        problem structure.  A string is looked up in the installed solver
        maps.  A :class:`Solver` instance is used directly (custom solver).
    problem_form : ProblemForm
        Pre-canonicalization structural analysis of the problem.
    gp : bool
        Whether the problem is a geometric program.

    Returns
    -------
    Solver
        A concrete solver instance.

    Raises
    ------
    SolverError
        If the solver argument is invalid, not installed, cannot handle
        the problem, or no suitable solver is found.
    """
    constant = len(problem_form._problem.variables()) == 0

    if isinstance(solver, Solver):
        if solver.name() in s.SOLVERS:
            raise SolverError(
                "Custom solvers must have a different name "
                "than the officially supported ones"
            )
        if not constant:
            if problem_form.is_mixed_integer() and not solver.MIP_CAPABLE:
                raise SolverError(
                    "Problem is mixed-integer, but the custom solver "
                    "%s is not MIP-capable." % solver.name()
                )
            if not solver.can_solve(problem_form):
                raise SolverError(
                    "The solver %s cannot solve this problem." % solver.name()
                )
        return solver

    if isinstance(solver, str):
        qp_inst, conic_inst = _lookup_by_name(solver, gp)

        if qp_inst is None and conic_inst is None:
            raise SolverError(
                "The solver %s is not installed." % solver.upper()
            )

        if gp and conic_inst is None:
            raise SolverError(
                "When `gp=True`, `solver` must be a conic solver "
                "(received '%s'); try calling "
                "`solve()` with `solver=cvxpy.CLARABEL`." % solver.upper()
            )

        if constant:
            # Constant problems are handled by ConstantSolver; any
            # installed solver is acceptable.
            return conic_inst or qp_inst  # type: ignore[return-value]

        # Prefer QP when the problem has a quadratic objective; fall back
        # to conic; try QP last for non-quadratic problems.
        candidates = []
        if qp_inst is not None and problem_form.has_quadratic_objective():
            candidates.append(qp_inst)
        if conic_inst is not None:
            candidates.append(conic_inst)
        if qp_inst is not None and not problem_form.has_quadratic_objective():
            candidates.append(qp_inst)

        for inst in candidates:
            if inst.can_solve(problem_form):
                return inst
        raise SolverError(
            "The solver %s cannot solve this problem." % solver.upper()
        )

    if solver is not None:
        raise SolverError(
            "The solver argument must be a string, a Solver instance, or None."
        )

    # solver is None
    if constant:
        # Constant problems are handled by ConstantSolver; return any
        # installed solver to satisfy the chain construction.
        for name in slv_def.INSTALLED_SOLVERS:
            for inst in (slv_def.SOLVER_MAP_CONIC.get(name),
                         slv_def.SOLVER_MAP_QP.get(name)):
                if inst is not None:
                    return inst

    default = pick_default_solver(problem_form)
    if default is not None:
        return default

    return _fallback_solver(problem_form)


def resolve_and_build_chain(
    problem: Problem,
    solver: str | Solver | None = None,
    gp: bool = False,
    enforce_dpp: bool = False,
    ignore_dpp: bool = False,
    canon_backend: str | None = None,
    solver_opts: dict | None = None,
) -> "SolvingChain":
    """Resolve a solver argument and build a solving chain.

    Resolves *solver* (``None``, a string name, or a :class:`Solver` instance)
    to a concrete solver, validates it against the problem structure, then
    delegates to :func:`_build_solving_chain`.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    solver : str, Solver, or None
        The solver to use. ``None`` selects a default based on problem
        structure. A string is looked up in the installed solver maps.
        A :class:`Solver` instance is used directly (custom solver).
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    enforce_dpp : bool
        When True, raise DPPError for non-DPP problems.
    ignore_dpp : bool
        When True, treat DPP problems as non-DPP.
    canon_backend : str or None
        Canonicalization backend (``'CPP'``, ``'SCIPY'``, or ``'COO'``).
    solver_opts : dict or None
        Solver-specific options.

    Returns
    -------
    SolvingChain
        A SolvingChain that can be used to solve the problem.

    Raises
    ------
    SolverError
        If no suitable solver is found or the specified solver cannot handle
        the problem.
    """
    problem_form = make_problem_form(problem, gp, ignore_dpp)
    solver_instance = _resolve_solver(solver, problem_form, gp)
    return _build_solving_chain(
        problem,
        solver_instance,
        problem_form=problem_form,
        gp=gp,
        enforce_dpp=enforce_dpp,
        ignore_dpp=ignore_dpp,
        canon_backend=canon_backend,
        solver_opts=solver_opts,
    )


def _validate_problem_data(data) -> None:
    """Validate problem data for NaN and Inf values.

    Raises ValueError if:
    - Any matrix/vector contains NaN
    - Objective or constraint matrix (not RHS/bounds) contains Inf

    Inf is allowed in constraint RHS (b, G) and bounds since users
    sometimes use inf for unbounded constraints/variables.
    """
    # Skip validation for non-dict data (e.g., ConstantSolver returns Problem)
    if not isinstance(data, dict):
        return

    # Keys where Inf is allowed (constraint RHS and variable bounds)
    inf_allowed_keys = {s.B, s.G, s.LOWER_BOUNDS, s.UPPER_BOUNDS}

    # Keys to check (objective coefficients, constraint matrices, bounds)
    keys_to_check = [s.P, s.C, s.Q, s.A, s.B, s.F, s.G,
                     s.LOWER_BOUNDS, s.UPPER_BOUNDS]

    for key in keys_to_check:
        if key not in data or data[key] is None:
            continue
        val = data[key]
        arr = val.data if sp.issparse(val) else val
        if key in inf_allowed_keys:
            if np.any(np.isnan(arr)):
                raise ValueError(
                    "Problem data contains NaN. "
                    "Check your parameter values and constants."
                )
        else:
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Problem data contains NaN or Inf. "
                    "Check your parameter values and constants."
                )


class SolvingChain(Chain):
    """A reduction chain that ends with a solver.

    Parameters
    ----------
    reductions : list[Reduction]
        A list of reductions. The last reduction in the list must be a solver
        instance.

    Attributes
    ----------
    reductions : list[Reduction]
        A list of reductions.
    solver : Solver
        The solver, i.e., reductions[-1].
    """

    def __init__(self, problem=None, reductions=None, solver_context=None) -> None:
        super(SolvingChain, self).__init__(problem=problem,
                                           reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError("Solving chains must terminate with a Solver.")
        self.solver = self.reductions[-1]
        self.solver_context = solver_context

    def prepend(self, chain) -> "SolvingChain":
        """
        Create and return a new SolvingChain by concatenating
        chain with this instance.
        """
        return SolvingChain(reductions=chain.reductions + self.reductions)

    def solve(self, problem, warm_start: bool, verbose: bool, solver_opts):
        """Solves the problem by applying the chain.

        Applies each reduction in the chain to the problem, solves it,
        and then inverts the chain to return a solution of the supplied
        problem.

        Parameters
        ----------
        problem : Problem
            The problem to solve.
        warm_start : bool
            Whether to warm start the solver.
        verbose : bool
            Whether to enable solver verbosity.
        solver_opts : dict
            Solver specific options.

        Returns
        -------
        solution : Solution
            A solution to the problem.
        """
        data, inverse_data = self.apply(problem)

        # We validate the data both in SolvingChain.solve and
        # in SolvingChain.solve_via_data. These are the two possible
        # entry points for executing the solving chain.
        _validate_problem_data(data)

        solution = self.solver.solve_via_data(data, warm_start,
                                              verbose, solver_opts)
        return self.invert(solution, inverse_data)

    def solve_via_data(self, problem, data, warm_start: bool = False, verbose: bool = False,
                       solver_opts=None):
        """Solves the problem using the data output by the an apply invocation.

        The semantics are:

        .. code :: python

            data, inverse_data = solving_chain.apply(problem)
            solution = solving_chain.invert(solver_chain.solve_via_data(data, ...))

        which is equivalent to writing

        .. code :: python

            solution = solving_chain.solve(problem, ...)

        Parameters
        ----------
        problem : Problem
            The problem to solve.
        data : map
            Data for the solver.
        warm_start : bool
            Whether to warm start the solver.
        verbose : bool
            Whether to enable solver verbosity.
        solver_opts : dict
            Solver specific options.

        Returns
        -------
        raw solver solution
            The information returned by the solver; this is not necessarily
            a Solution object.
        """
        if solver_opts is None:
            solver_opts = {}

        # We validate the data both in SolvingChain.solve and
        # in SolvingChain.solve_via_data. These are the two possible
        # entry points for executing the solving chain.
        _validate_problem_data(data)

        return self.solver.solve_via_data(data, warm_start, verbose,
                                          solver_opts, problem._solver_cache)
