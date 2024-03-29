from __future__ import annotations

import warnings

import numpy as np

from cvxpy.atoms import EXP_ATOMS, NONPOS_ATOMS, PSD_ATOMS, SOC_ATOMS
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
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import (
    Valinvec2mixedint,
)
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import CLARABEL, ECOS, PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg

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
    Please either add an explicit dependency on ECOS or switch to our new
    default solver, Clarabel, by either not specifying a solver argument
    or specifying ``solver=cp.CLARABEL``.
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


def _solve_as_qp(problem, candidates):
    if _is_lp(problem) and \
            [s for s in candidates['conic_solvers'] if s not in candidates['qp_solvers']]:
        # OSQP can take many iterations for LPs; use a conic solver instead
        # GUROBI and CPLEX QP/LP interfaces are more efficient
        #   -> Use them instead of conic if applicable.
        return False
    return candidates['qp_solvers'] and qp2symbolic_qp.accepts(problem)


def _reductions_for_problem_class(problem, candidates, gp: bool = False, solver_opts=None) \
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

    # Dcp2Cone and Qp2SymbolicQp require problems to minimize their objectives.
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]

    # Special reduction for finite set constraint, 
    # used by both QP and conic pathways.
    constr_types = {type(c) for c in problem.constraints}
    if FiniteSet in constr_types:
        reductions += [Valinvec2mixedint()]

    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    valid_qp = _solve_as_qp(problem, candidates) and use_quad
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
    reductions = _reductions_for_problem_class(problem, candidates, gp, solver_opts)

    # Process DPP status of the problem.
    dpp_context = 'dcp' if not gp else 'dgp'
    if ignore_dpp or not problem.is_dpp(dpp_context):
        # No warning for ignore_dpp.
        if ignore_dpp:
            reductions = [EvalParams()] + reductions
        elif not enforce_dpp:
            warnings.warn(DPP_ERROR_MSG)
            reductions = [EvalParams()] + reductions
        else:
            raise DPPError(DPP_ERROR_MSG)
    elif any(param.is_complex() for param in problem.parameters()):
        reductions = [EvalParams()] + reductions
    else:  # Compilation with DPP.
        n_parameters = sum(np.prod(param.shape) for param in problem.parameters())
        if n_parameters >= PARAM_THRESHOLD:
            warnings.warn(
                "Your problem has too many parameters for efficient DPP "
                "compilation. We suggest setting 'ignore_dpp = True'."
            )

    # Conclude with matrix stuffing; choose one of the following paths:
    #   (1) QpMatrixStuffing --> [a QpSolver],
    #   (2) ConeMatrixStuffing --> [a ConicSolver]
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    if _solve_as_qp(problem, candidates) and use_quad:
        # Canonicalize as a QP
        solver = candidates['qp_solvers'][0]
        solver_instance = slv_def.SOLVER_MAP_QP[solver]
        reductions += [
            CvxAttr2Constr(reduce_bounds=not solver_instance.BOUNDED_VARIABLES), 
            qp2symbolic_qp.Qp2SymbolicQp(),
            QpMatrixStuffing(canon_backend=canon_backend),
            solver_instance,
        ]
        return SolvingChain(reductions=reductions)

    # Canonicalize as a cone program
    if not candidates['conic_solvers']:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)

    constr_types = set()
    # ^ We use constr_types to infer an incomplete list of cones that
    # the solver will need after canonicalization.
    for c in problem.constraints:
        constr_types.add(type(c))
    ex_cos = [ct for ct in constr_types if ct in EXOTIC_CONES]
    approx_cos = [ct for ct in constr_types if ct in APPROX_CONES]
    # ^ The way we populate "ex_cos" will need to change if and when
    # we have atoms that require exotic cones.
    for co in ex_cos:
        sim_cos = EXOTIC_CONES[co]  # get the set of required simple cones
        constr_types.update(sim_cos)
        constr_types.remove(co)

    for co in approx_cos:
        app_cos = APPROX_CONES[co]
        constr_types.update(app_cos)
        constr_types.remove(co)
    # We now go over individual elementary cones support by CVXPY (
    # SOC, ExpCone, NonNeg, Zero, PSD, PowCone3D) and check if
    # they've appeared in constr_types or if the problem has an atom
    # requiring that cone.
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
    if PowCone3D in constr_types:
        # if we add in atoms that specifically use the 3D power cone
        # (rather than the ND power cone), then we'll need to check
        # for those atoms here as well.
        cones.append(PowCone3D)

    # Here, we make use of the observation that canonicalization only
    # increases the number of constraints in our problem.
    var_domains = sum([var.domain for var in problem.variables()], start = [])
    has_constr = len(cones) > 0 or len(problem.constraints) > 0 or len(var_domains) > 0

    if PSD in cones \
            and slv_def.DISREGARD_CLARABEL_SDP_SUPPORT_FOR_DEFAULT_RESOLUTION \
            and specified_solver is None:
        candidates['conic_solvers'] = [s for s in candidates['conic_solvers'] if s != CLARABEL]

    for solver in candidates['conic_solvers']:
        solver_instance = slv_def.SOLVER_MAP_CONIC[solver]
        # Cones supported for MI problems may differ from non MI.
        if problem.is_mixed_integer():
            supported_constraints = solver_instance.MI_SUPPORTED_CONSTRAINTS
        else:
            supported_constraints = solver_instance.SUPPORTED_CONSTRAINTS
        unsupported_constraints = [
            cone for cone in cones if cone not in supported_constraints
        ]

        if has_constr or not solver_instance.REQUIRES_CONSTR:
            if ex_cos:
                reductions.append(Exotic2Common())
            if RelEntrConeQuad in approx_cos or OpRelEntrConeQuad in approx_cos:
                reductions.append(QuadApprox())

            # Should the objective be canonicalized to a quadratic?
            if solver_opts is None:
                use_quad_obj = True
            else:
                use_quad_obj = solver_opts.get("use_quad_obj", True)
            quad_obj = use_quad_obj and solver_instance.supports_quad_obj() and \
                problem.objective.expr.has_quadratic_term()
            reductions += [
                Dcp2Cone(quad_obj=quad_obj),
                CvxAttr2Constr(reduce_bounds=not solver_instance.BOUNDED_VARIABLES),
            ]
            if all(c in supported_constraints for c in cones):
                if solver == ECOS and specified_solver == ECOS:
                    warnings.warn(ECOS_DEP_DEPRECATION_MSG, FutureWarning)
                elif solver == ECOS and specified_solver is None:
                    warnings.warn(ECOS_DEPRECATION_MSG, FutureWarning)
                # Return the reduction chain.
                reductions += [
                    ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend),
                    solver_instance
                ]
                return SolvingChain(reductions=reductions)
            elif all(c==SOC for c in unsupported_constraints) and PSD in supported_constraints:
                reductions += [
                    SOC2PSD(),
                    ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend),
                    solver_instance
                ]
                return SolvingChain(reductions=reductions)

    raise SolverError("Either candidate conic solvers (%s) do not support the "
                      "cones output by the problem (%s), or there are not "
                      "enough constraints in the problem." % (
                          candidates['conic_solvers'],
                          ", ".join([cone.__name__ for cone in cones])))


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

    def __init__(self, problem=None, reductions=None) -> None:
        super(SolvingChain, self).__init__(problem=problem,
                                           reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError("Solving chains must terminate with a Solver.")
        self.solver = self.reductions[-1]

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
        solution = self.solver.solve_via_data(data, warm_start,
                                              verbose, solver_opts)
        return self.invert(solution, inverse_data)

    def solve_via_data(self, problem, data, warm_start: bool = False, verbose: bool = False,
                       solver_opts={}):
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
        return self.solver.solve_via_data(data, warm_start, verbose,
                                          solver_opts, problem._solver_cache)
