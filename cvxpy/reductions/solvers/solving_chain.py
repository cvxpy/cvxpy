from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.atoms import (
    EXP_ATOMS,
    NONPOS_ATOMS,
    POWCONE_ATOMS,
    POWCONE_ND_ATOMS,
    PSD_ATOMS,
    SOC_ATOMS,
)
from cvxpy.constraints import (
    PSD,
    SOC,
    Equality,
    ExpCone,
    FiniteSet,
    Inequality,
    NonNeg,
    NonPos,
    PowCone3D,
    PowConeND,
    Zero,
)
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approximations import APPROX_CONES, QuadApprox
from cvxpy.reductions.cone2cone.exotic2common import (
    EXOTIC_CONES,
    Exotic2Common,
)
from cvxpy.reductions.cone2cone.soc2psd import SOC2PSD
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
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import CLARABEL, COO_CANON_BACKEND, DPP_PARAM_THRESHOLD
from cvxpy.utilities import scopes
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

ECOS_DEP_DEPRECATION_MSG = (
    """
    You specified your problem should be solved by ECOS. Starting in
    CXVPY 1.6.0, ECOS will no longer be installed by default with CVXPY.
    Please either add ECOS as an explicit install dependency to your project
    or switch to our new default solver, Clarabel, by either not specifying a
    solver argument or specifying ``solver=cp.CLARABEL``. To suppress this
    warning while continuing to use ECOS, you can filter this warning using
    Python's ``warnings`` module until you are using 1.6.0.
    """
)

ECOS_DEPRECATION_MSG = (
    """
    Your problem is being solved with the ECOS solver by default. Starting in
    CVXPY 1.5.0, Clarabel will be used as the default solver instead. To continue
    using ECOS, specify the ECOS solver explicitly using the ``solver=cp.ECOS``
    argument to the ``problem.solve`` method.
    """
)


def _is_lp(self):
    """Is problem a linear program?
    """
    for c in self.constraints:
        if not (isinstance(c, (Equality, Zero)) or c.args[0].is_pwl()):
            return False
    for var in self.variables():
        if var.is_psd() or var.is_nsd():
            return False
    return (self.is_dcp() and self.objective.args[0].is_pwl())


def _check_dpp(problem, gp: bool, enforce_dpp: bool, ignore_dpp: bool,
               solver_supports_quad_obj: bool) -> bool:
    """Check if problem is DPP and handle warnings/errors.

    Returns True if DPP compilation should be used, False otherwise.
    """
    if ignore_dpp:
        return False

    # Determine DPP context
    dpp_context = 'dgp' if gp else 'dcp'

    # For QP/conic-QP solvers, we can loosen the DPP rules for quad_form.
    # These solvers accept quadratic objectives directly (P matrix), so the
    # mapping from parameters to problem data (P, q) stays linear — the
    # standard DPP requirement. P can remain parametric since its value is
    # only needed at solve time, not at canonicalization time.
    if solver_supports_quad_obj:
        with scopes.quad_form_dpp_scope():
            is_dpp = problem.is_dpp(dpp_context)
    else:
        is_dpp = problem.is_dpp(dpp_context)

    if not is_dpp:
        if enforce_dpp:
            raise DPPError(DPP_ERROR_MSG)
        warn(DPP_ERROR_MSG)
        return False

    return True


def _select_canon_backend(problem, canon_backend: str | None, use_dpp: bool) -> str | None:
    """Select canon backend for best performance.

    For DPP with many parameters, auto-select COO backend.
    Validation/fallback happens later in ConeMatrixStuffing.
    """
    if canon_backend is not None:
        return canon_backend
    if use_dpp:
        total_param_size = sum(p.size for p in problem.parameters())
        if total_param_size >= DPP_PARAM_THRESHOLD:
            return COO_CANON_BACKEND
    return None


def _solve_as_qp(problem, candidates, ignore_dpp: bool = False):
    if _is_lp(problem) and \
            [s for s in candidates['conic_solvers'] if s not in candidates['qp_solvers']]:
        # OSQP can take many iterations for LPs; use a conic solver instead
        # GUROBI and CPLEX QP/LP interfaces are more efficient
        #   -> Use them instead of conic if applicable.
        return False
    # For DPP problems with parameters, check is_qp in DPP scope
    # because canonicalization will preserve parameters as non-constant
    if not ignore_dpp and problem.parameters() and problem.is_dpp():
        with scopes.dpp_scope():
            return candidates['qp_solvers'] and problem.is_qp()
    return candidates['qp_solvers'] and problem.is_qp()


def _find_conic_solver(problem, candidates, solver_opts, specified_solver):
    """Find a compatible conic solver for the problem.

    Returns a dict with solver info needed for building reductions:
        - solver_instance: the solver
        - solver_context: SolverInfo for the solver
        - quad_obj: whether to use quadratic objective
        - approx_cos: list of approximate cones
        - ex_cos: set of exotic cones to convert
        - use_soc2psd: whether to convert SOC to PSD
        - use_soc_dim3: whether to convert SOC to dim-3

    Raises SolverError if no compatible solver found.
    """
    # Determine required cones from problem
    constr_types = set()
    for c in problem.constraints:
        constr_types.add(type(c))
    approx_cos = [ct for ct in constr_types if ct in APPROX_CONES]

    for co in approx_cos:
        app_cos = APPROX_CONES[co]
        constr_types.update(app_cos)
        constr_types.remove(co)

    cones = []
    atoms = problem.atoms()

    if SOC in constr_types or any(atom in SOC_ATOMS for atom in atoms):
        cones.append(SOC)
    if ExpCone in constr_types or any(atom in EXP_ATOMS for atom in atoms):
        cones.append(ExpCone)
    if any(t in constr_types for t in [Inequality, NonPos, NonNeg]) \
            or any(atom in NONPOS_ATOMS for atom in atoms):
        cones.append(NonNeg)
    if Equality in constr_types or Zero in constr_types:
        cones.append(Zero)
    if PSD in constr_types \
            or any(atom in PSD_ATOMS for atom in atoms) \
            or any(v.is_psd() or v.is_nsd() for v in problem.variables()):
        cones.append(PSD)
    if PowCone3D in constr_types or any(atom in POWCONE_ATOMS for atom in atoms):
        cones.append(PowCone3D)
    if PowConeND in constr_types or any(atom in POWCONE_ND_ATOMS for atom in atoms):
        cones.append(PowConeND)

    var_domains = sum([var.domain for var in problem.variables()], start=[])
    has_constr = len(cones) > 0 or len(problem.constraints) > 0 or len(var_domains) > 0

    conic_candidates = candidates['conic_solvers']
    if PSD in cones \
            and slv_def.DISREGARD_CLARABEL_SDP_SUPPORT_FOR_DEFAULT_RESOLUTION \
            and specified_solver is None:
        conic_candidates = [s for s in conic_candidates if s != CLARABEL]

    # Find first compatible solver
    for solver_name in conic_candidates:
        solver_instance = slv_def.SOLVER_MAP_CONIC[solver_name]
        if problem.is_mixed_integer():
            supported_constraints = solver_instance.MI_SUPPORTED_CONSTRAINTS
        else:
            supported_constraints = solver_instance.SUPPORTED_CONSTRAINTS

        solver_context = SolverInfo(
            solver=solver_name,
            supported_constraints=supported_constraints,
            supports_bounds=solver_instance.BOUNDED_VARIABLES
        )

        cones_set = set(cones)
        ex_cos = (cones_set & set(EXOTIC_CONES)) - set(supported_constraints)

        for co in ex_cos:
            sim_cos = set(EXOTIC_CONES[co])
            cones_set.update(sim_cos)
            cones_set.discard(co)

        unsupported = [c for c in cones_set if c not in supported_constraints]

        if not (has_constr or not solver_instance.REQUIRES_CONSTR):
            continue

        use_quad_obj = True if solver_opts is None else solver_opts.get("use_quad_obj", True)
        quad_obj = (use_quad_obj and solver_instance.supports_quad_obj() and
                    problem.objective.expr.has_quadratic_term())

        if all(c in supported_constraints for c in cones_set):
            return {
                'solver_instance': solver_instance,
                'solver_context': solver_context,
                'quad_obj': quad_obj,
                'approx_cos': approx_cos,
                'ex_cos': ex_cos,
                'use_soc2psd': False,
                'use_soc_dim3': solver_instance.SOC_DIM3_ONLY and SOC in cones_set,
            }
        elif all(c == SOC for c in unsupported) and PSD in supported_constraints:
            return {
                'solver_instance': solver_instance,
                'solver_context': solver_context,
                'quad_obj': quad_obj,
                'approx_cos': approx_cos,
                'ex_cos': ex_cos,
                'use_soc2psd': True,
                'use_soc_dim3': False,
            }

    raise SolverError("Either candidate conic solvers (%s) do not support the "
                      "cones output by the problem (%s), or there are not "
                      "enough constraints in the problem." % (
                          candidates['conic_solvers'],
                          ", ".join([cone.__name__ for cone in cones])))


def _reductions_for_problem_class(problem, candidates, gp: bool = False,
                                   ignore_dpp: bool = False, solver_opts=None) \
        -> list[Reduction]:
    """
    Builds a chain that rewrites a problem into an intermediate
    representation suitable for numeric reductions.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    candidates : dict
        Dictionary of candidate solvers divided in qp_solvers
        and conic_solvers.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    ignore_dpp : bool
        If True, DPP analysis is skipped when checking problem type.
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

    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    valid_qp = _solve_as_qp(problem, candidates, ignore_dpp) and use_quad
    valid_conic = len(candidates['conic_solvers']) > 0
    if not valid_qp and not valid_conic:
        raise SolverError("Problem could not be reduced to a QP, and no "
                            "conic solvers exist among candidate solvers "
                            "(%s)." % candidates)

    return reductions


def construct_solving_chain(problem, candidates,
                            gp: bool = False,
                            enforce_dpp: bool = False,
                            ignore_dpp: bool = False,
                            canon_backend: str | None = None,
                            solver_opts: dict | None = None,
                            specified_solver: str | None = None,
                            ) -> "SolvingChain":
    """Build a reduction chain from a problem to an installed solver.

    Note that if the supplied problem has 0 variables, then the solver
    parameter will be ignored.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    candidates : dict
        Dictionary of candidate solvers divided in qp_solvers
        and conic_solvers.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    enforce_dpp : bool, optional
        When True, a DPPError will be thrown when trying to parse a non-DPP
        problem (instead of just a warning). Defaults to False.
    ignore_dpp : bool, optional
        When True, DPP problems will be treated as non-DPP,
        which may speed up compilation. Defaults to False.
    canon_backend : str, optional
        'CPP' (default) | 'SCIPY'
        Specifies which backend to use for canonicalization, which can affect
        compilation time. Defaults to None, i.e., selecting the default
        backend.
    solver_opts : dict, optional
        Additional arguments to pass to the solver.
    specified_solver: str, optional
        A solver specified by the user.

    Returns
    -------
    SolvingChain
        A SolvingChain that can be used to solve the problem.

    Raises
    ------
    SolverError
        Raised if no suitable solver exists among the installed solvers, or
        if the target solver is not installed.
    """
    if len(problem.variables()) == 0:
        return SolvingChain(reductions=[ConstantSolver()])

    reductions = _reductions_for_problem_class(problem, candidates, gp, ignore_dpp,
                                               solver_opts)

    # Step 1: Select solver
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    is_qp = _solve_as_qp(problem, candidates, ignore_dpp) and use_quad

    if is_qp:
        solver_instance = slv_def.SOLVER_MAP_QP[candidates['qp_solvers'][0]]
        solver_supports_quad_obj = True
        solver_context = None
    else:
        if not candidates['conic_solvers']:
            raise SolverError("Problem could not be reduced to a QP, and no "
                              "conic solvers exist among candidate solvers "
                              "(%s)." % candidates)
        conic_info = _find_conic_solver(problem, candidates, solver_opts, specified_solver)
        solver_instance = conic_info['solver_instance']
        solver_context = conic_info['solver_context']
        solver_supports_quad_obj = solver_instance.supports_quad_obj()

    # Step 2: Handle DPP
    use_dpp = _check_dpp(problem, gp, enforce_dpp, ignore_dpp, solver_supports_quad_obj)
    if not use_dpp:
        reductions = [EvalParams()] + reductions
    else:
        canon_backend = _select_canon_backend(problem, canon_backend, use_dpp)

    # Step 3: Build reductions for the selected solver
    if is_qp:
        reductions += [
            Dcp2Cone(quad_obj=True),
            CvxAttr2Constr(reduce_bounds=not solver_instance.BOUNDED_VARIABLES),
            EliminateZeroSized(),
            ConeMatrixStuffing(quad_obj=True, canon_backend=canon_backend),
            solver_instance,
        ]
        return SolvingChain(reductions=reductions)
    else:
        # Conic path
        quad_obj = conic_info['quad_obj']
        approx_cos = conic_info['approx_cos']
        ex_cos = conic_info['ex_cos']

        if RelEntrConeQuad in approx_cos or OpRelEntrConeQuad in approx_cos:
            reductions.append(QuadApprox())

        reductions.append(Dcp2Cone(quad_obj=quad_obj, solver_context=solver_context))

        if ex_cos:
            reductions.append(Exotic2Common())

        reductions.append(CvxAttr2Constr(reduce_bounds=not solver_instance.BOUNDED_VARIABLES))
        reductions.append(EliminateZeroSized())

        if conic_info['use_soc_dim3']:
            reductions.append(SOCDim3())
        if conic_info['use_soc2psd']:
            reductions.append(SOC2PSD())

        reductions += [
            ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend),
            solver_instance
        ]
        return SolvingChain(reductions=reductions, solver_context=solver_context)


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
