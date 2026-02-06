from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import (
    SOC,
    FiniteSet,
)
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem_form import ProblemForm, pick_default_solver
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approx import (
    APPROX_CONE_CONVERSIONS,
    ApproxCone2Cone,
)
from cvxpy.reductions.cone2cone.exact import (
    EXACT_CONE_CONVERSIONS,
    ExactCone2Cone,
)
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
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import COO_CANON_BACKEND, DPP_PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.warn import warn

DPP_ERROR_MSG = (
    "You are solving a parameterized problem that is not DPP. "
    "Because the problem is not DPP, subsequent solves will not be "
    "faster than the first one. For more information, see the "
    "documentation on Disciplined Parametrized Programming, at "
    "https://www.cvxpy.org/tutorial/dpp/index.html"
)

def _pre_canonicalization_reductions(problem, gp: bool = False) \
        -> list[Reduction]:
    """
    Builds reductions that must run before DCP/DGP canonicalization.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    Returns
    -------
    list of Reduction objects
        A list of reductions that can be used to convert the problem to an
        intermediate form.
    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True.
    """
    reductions = []
    # TODO Handle boolean constraints.
    if complex2real.accepts(problem):
        reductions += [complex2real.Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]

    if not gp and not problem.is_dcp():
        append = build_non_disciplined_error_msg(problem, 'DCP')
        if problem.is_dgp():
            append += ("\nHowever, the problem does follow DGP rules. "
                       "Consider calling solve() with `gp=True`.")
        elif problem.is_dqcp():
            append += ("\nHowever, the problem does follow DQCP rules. "
                       "Consider calling solve() with `qcp=True`.")
        raise DCPError(
            "Problem does not follow DCP rules. Specifically:\n" + append)
    elif gp and not problem.is_dgp():
        append = build_non_disciplined_error_msg(problem, 'DGP')
        if problem.is_dcp():
            append += ("\nHowever, the problem does follow DCP rules. "
                       "Consider calling solve() with `gp=False`.")
        elif problem.is_dqcp():
            append += ("\nHowever, the problem does follow DQCP rules. "
                       "Consider calling solve() with `qcp=True`.")
        raise DGPError("Problem does not follow DGP rules." + append)

    # Dcp2Cone requires problems to minimize their objectives.
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]

    # Special reduction for finite set constraint,
    # used by both QP and conic pathways.
    constr_types = {type(c) for c in problem.constraints}
    if FiniteSet in constr_types:
        reductions += [Valinvec2mixedint()]

    return reductions


def build_solving_chain(
    problem,
    solver_instance: Solver,
    problem_form=None,
    gp: bool = False,
    enforce_dpp: bool = False,
    ignore_dpp: bool = False,
    canon_backend: str | None = None,
    solver_opts: dict | None = None,
) -> "SolvingChain":
    """Build a reduction chain for a specific solver.

    Builds the chain directly for the given solver instance.

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

    reductions = _pre_canonicalization_reductions(problem, gp)

    # --- DPP handling ---
    dpp_context = 'dcp' if not gp else 'dgp'
    if ignore_dpp or not problem.is_dpp(dpp_context):
        if ignore_dpp:
            reductions = [EvalParams()] + reductions
        elif not enforce_dpp:
            warn(DPP_ERROR_MSG)
            reductions = [EvalParams()] + reductions
        else:
            raise DPPError(DPP_ERROR_MSG)
    else:
        if canon_backend is None:
            total_param_size = sum(p.size for p in problem.parameters())
            if total_param_size >= DPP_PARAM_THRESHOLD:
                canon_backend = COO_CANON_BACKEND

    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)

    # --- Build reduction chain ---
    if problem.is_mixed_integer():
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

    # Determine cone conversions needed.
    # QP solvers always need quad_obj=True in the matrix stuffing step
    # because their apply() expects the P matrix from ConeMatrixStuffing.
    is_qp_solver = isinstance(solver_instance, QpSolver)
    quad_obj = (use_quad and solver_instance.supports_quad_obj()
                and (is_qp_solver or problem_form.has_quadratic_objective()))
    cones = problem_form.cones(quad_obj=quad_obj).copy()

    exact_targets = (cones & EXACT_CONE_CONVERSIONS.keys()) - supported
    for co in exact_targets:
        cones.discard(co)
        cones.update(EXACT_CONE_CONVERSIONS[co])

    approx_cos = (cones & APPROX_CONE_CONVERSIONS.keys()) - supported
    for co in approx_cos:
        cones.discard(co)
        cones.update(APPROX_CONE_CONVERSIONS[co])

    reductions.append(Dcp2Cone(quad_obj=quad_obj, solver_context=solver_context))

    if exact_targets:
        reductions.append(ExactCone2Cone(target_cones=exact_targets))
    if approx_cos:
        reductions.append(ApproxCone2Cone(target_cones=approx_cos))

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


def resolve_and_build_chain(
    problem,
    solver=None,
    gp: bool = False,
    enforce_dpp: bool = False,
    ignore_dpp: bool = False,
    canon_backend: str | None = None,
    solver_opts: dict | None = None,
) -> "SolvingChain":
    """Resolve a solver argument and build a solving chain.

    Resolves *solver* (``None``, a string name, or a :class:`Solver` instance)
    to a concrete solver, validates it against the problem structure, then
    delegates to :func:`build_solving_chain`.

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
    # Validate DCP/DGP compliance before solver resolution.
    # This must happen first so that non-DCP problems raise DCPError
    # (not SolverError from can_solve failing on ill-formed cones).
    if not gp and not problem.is_dcp():
        append = build_non_disciplined_error_msg(problem, 'DCP')
        if problem.is_dgp():
            append += ("\nHowever, the problem does follow DGP rules. "
                       "Consider calling solve() with `gp=True`.")
        elif problem.is_dqcp():
            append += ("\nHowever, the problem does follow DQCP rules. "
                       "Consider calling solve() with `qcp=True`.")
        raise DCPError(
            "Problem does not follow DCP rules. Specifically:\n" + append)
    elif gp and not problem.is_dgp():
        append = build_non_disciplined_error_msg(problem, 'DGP')
        if problem.is_dcp():
            append += ("\nHowever, the problem does follow DCP rules. "
                       "Consider calling solve() with `gp=False`.")
        elif problem.is_dqcp():
            append += ("\nHowever, the problem does follow DQCP rules. "
                       "Consider calling solve() with `qcp=True`.")
        raise DGPError("Problem does not follow DGP rules." + append)

    # Quick validation: reject unknown or uninstalled solver names
    # before the zero-variable early return, so that e.g.
    # Problem(Minimize(0)).solve(solver='DAQP') still raises SolverError
    # when DAQP is not installed.
    if isinstance(solver, str):
        _upper = solver.upper()
        _qp = slv_def.SOLVER_MAP_QP.get(_upper)
        _co = slv_def.SOLVER_MAP_CONIC.get(_upper)
        if _qp is None and _co is None:
            raise SolverError("The solver %s is not installed." % _upper)
        if not (_qp is not None and _qp.is_installed()) \
                and not (_co is not None and _co.is_installed()):
            raise SolverError("The solver %s is not installed." % _upper)
    elif isinstance(solver, Solver):
        if solver.name() in s.SOLVERS:
            raise SolverError(
                "Custom solvers must have a different name "
                "than the officially supported ones"
            )
    elif solver is not None:
        raise SolverError(
            "The solver argument must be a string, a Solver instance, or None."
        )

    # Zero-variable problems are handled directly by ConstantSolver,
    # no solver resolution needed.
    if len(problem.variables()) == 0:
        return SolvingChain(reductions=[ConstantSolver()])

    # When ignore_dpp is set and the problem has parameters, EvalParams
    # will evaluate all parameters to constants before canonicalization.
    # Tell ProblemForm so it can exclude parameter-only sub-expressions
    # from cone detection (they won't need conic canonicalization).
    eval_params = ignore_dpp and bool(problem.parameters())
    problem_form = ProblemForm(problem, gp=gp, eval_params=eval_params)

    if isinstance(solver, Solver):
        # --- Custom solver instance ---
        # (name-vs-SOLVERS check already done above)
        if problem_form.is_mixed_integer() and not solver.MIP_CAPABLE:
            raise SolverError(
                "Problem is mixed-integer, but the custom solver "
                "%s is not MIP-capable." % solver.name()
            )
        if not solver.can_solve(problem_form):
            raise SolverError(
                "The solver %s cannot solve this problem." % solver.name()
            )
        solver_instance = solver

    elif isinstance(solver, str):
        # --- Named solver ---
        # (existence and is_installed checks already done above;
        #  recompute installed instances for QP/conic routing)
        solver_upper = solver.upper()
        qp_inst = slv_def.SOLVER_MAP_QP.get(solver_upper)
        conic_inst = slv_def.SOLVER_MAP_CONIC.get(solver_upper)
        if qp_inst is not None and not qp_inst.is_installed():
            qp_inst = None
        if conic_inst is not None and not conic_inst.is_installed():
            conic_inst = None

        # GP problems must use conic solvers (no QP path).
        if gp and conic_inst is None:
            raise SolverError(
                "When `gp=True`, `solver` must be a conic solver "
                "(received '%s'); try calling "
                "`solve()` with `solver=cvxpy.CLARABEL`." % solver_upper
            )
        if gp:
            qp_inst = None

        # When the solver appears in both maps, prefer QP when the problem
        # has a quadratic objective and the QP instance can handle it.
        if qp_inst is not None and conic_inst is not None:
            if (problem_form.has_quadratic_objective()
                    and qp_inst.can_solve(problem_form)):
                solver_instance = qp_inst
            elif conic_inst.can_solve(problem_form):
                solver_instance = conic_inst
            else:
                raise SolverError(
                    "The solver %s cannot solve this problem." % solver_upper
                )
        elif qp_inst is not None:
            if not qp_inst.can_solve(problem_form):
                raise SolverError(
                    "The solver %s cannot solve this problem." % solver_upper
                )
            solver_instance = qp_inst
        else:
            assert conic_inst is not None  # guaranteed by the None-check above
            if not conic_inst.can_solve(problem_form):
                raise SolverError(
                    "The solver %s cannot solve this problem." % solver_upper
                )
            solver_instance = conic_inst

    elif solver is None:
        # --- Default solver selection ---
        if problem_form.is_mixed_integer() and not problem.is_lp():
            warn(
                "Your problem is mixed-integer but not an LP. "
                "If your problem is nonlinear, consider installing SCIP "
                "(pip install pyscipopt) to solve it."
            )
        default = pick_default_solver(problem_form)
        if default is not None:
            solver_instance = default
        else:
            # Fallback: iterate all installed solvers.
            solver_instance = None
            for inst in list(slv_def.SOLVER_MAP_CONIC.values()) + \
                    list(slv_def.SOLVER_MAP_QP.values()):
                if inst.is_installed() and inst.can_solve(problem_form):
                    solver_instance = inst
                    warn(
                        "The default solvers are not available. "
                        "Using %s. Consider specifying a solver "
                        "explicitly via the ``solver`` argument." %
                        inst.name()
                    )
                    break
            if solver_instance is None:
                raise SolverError(
                    "No installed solver could handle this problem."
                )
    else:
        assert False, "unreachable: early validation rejects invalid solver types"

    return build_solving_chain(
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
