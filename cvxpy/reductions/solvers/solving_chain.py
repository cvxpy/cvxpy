from cvxpy.atoms import EXP_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import ExpCone, PSD, SOC
from cvxpy.error import DCPError, DGPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions import (Chain, ConeMatrixStuffing, Dcp2Cone, EvalParams,
                              FlipObjective, Dgp2Dcp, Qp2SymbolicQp, QpMatrixStuffing,
                              CvxAttr2Constr, Complex2Real)
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.defines import (SOLVER_MAP_CONIC,
                                              SOLVER_MAP_QP,
                                              INSTALLED_SOLVERS,
                                              CONIC_SOLVERS,
                                              QP_SOLVERS)


def construct_solving_chain(problem, solver=None, gp=False):
    """Build a reduction chain from a problem to an installed solver.

    Note that if the supplied problem has 0 variables, then the solver
    parameter will be ignored.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    solver : string
        The name of the solver with which to terminate the chain. If no solver
        is supplied (i.e., if solver is None), then the targeted solver may be
        any of those that are installed. If the problem is variable-free,
        then this parameter is ignored.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.

    Returns
    -------
    SolvingChain
        A SolvingChain that can be used to solve the problem.

    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True
    SolverError
        Raised if no suitable solver exists among the installed solvers, or
        if the target solver is not installed.
    """
    if solver is not None:
        if solver not in INSTALLED_SOLVERS:
            raise SolverError("The solver %s is not installed." % solver)
        candidates = [solver]
    else:
        candidates = INSTALLED_SOLVERS

    reductions = []
    if problem.parameters():
        reductions += [EvalParams()]
    if len(problem.variables()) == 0:
        reductions += [ConstantSolver()]
        return SolvingChain(reductions=reductions)
    if Complex2Real().accepts(problem):
        reductions += [Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]

    if not gp and not problem.is_dcp():
        append = ""
        if problem.is_dgp():
            append = (" However, the problem does follow DGP rules. "
                      "Consider calling this function with `gp=True`.")
        raise DCPError("Problem does not follow DCP rules." + append)
    elif gp and not problem.is_dgp():
        append = ""
        if problem.is_dcp():
            append = (" However, the problem does follow DCP rules. "
                      "Consider calling this function with `gp=False`.")
        raise DGPError("Problem does not follow DGP rules." + append)

    # Dcp2Cone and Qp2SymbolicQp require problems to minimize their objectives.
    if type(problem.objective) == Maximize:
        reductions.append(FlipObjective())

    # Conclude the chain with one of the following:
    #   (1) Qp2SymbolicQp --> QpMatrixStuffing --> [a QpSolver],
    #   (2) Dcp2Cone --> ConeMatrixStuffing --> [a ConicSolver]
    #
    # First, attempt to canonicalize the problem to a linearly constrained QP.
    candidate_qp_solvers = [s for s in QP_SOLVERS if s in candidates]
    # Consider only MIQP solvers if problem is integer
    if problem.is_mixed_integer():
        candidate_qp_solvers = \
            [s for s in candidate_qp_solvers if
             SOLVER_MAP_QP[s].MIP_CAPABLE]
    if candidate_qp_solvers and Qp2SymbolicQp().accepts(problem):
        solver = sorted(candidate_qp_solvers,
                        key=lambda s: QP_SOLVERS.index(s))[0]
        solver_instance = SOLVER_MAP_QP[solver]
        reductions += [CvxAttr2Constr(),
                       Qp2SymbolicQp(),
                       QpMatrixStuffing(),
                       solver_instance]
        return SolvingChain(reductions=reductions)

    candidate_conic_solvers = [s for s in CONIC_SOLVERS if s in candidates]
    if problem.is_mixed_integer():
        candidate_conic_solvers = \
            [s for s in candidate_conic_solvers if
             SOLVER_MAP_CONIC[s].MIP_CAPABLE]
        if not candidate_conic_solvers and \
                not candidate_qp_solvers:
            raise SolverError("Problem is mixed-integer, but candidate "
                              "QP/Conic solvers (%s) are not MIP-capable." %
                              [candidate_qp_solvers, candidate_conic_solvers])
    if not candidate_conic_solvers:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)

    # Attempt to canonicalize the problem to a cone program.
    # Our choice of solver depends upon which atoms are present in the
    # problem. The types of atoms to check for are SOC atoms, PSD atoms,
    # and exponential atoms.
    atoms = problem.atoms()
    cones = []
    if (any(atom in SOC_ATOMS for atom in atoms)
            or any(type(c) == SOC for c in problem.constraints)):
        cones.append(SOC)
    if (any(atom in EXP_ATOMS for atom in atoms)
            or any(type(c) == ExpCone for c in problem.constraints)):
        cones.append(ExpCone)
    if (any(atom in PSD_ATOMS for atom in atoms)
            or any(type(c) == PSD for c in problem.constraints)
            or any(v.is_psd() or v.is_nsd()
                   for v in problem.variables())):
        cones.append(PSD)

    # Here, we make use of the observation that canonicalization only
    # increases the number of constraints in our problem.
    has_constr = len(cones) > 0 or len(problem.constraints) > 0

    for solver in sorted(candidate_conic_solvers,
                         key=lambda s: CONIC_SOLVERS.index(s)):
        solver_instance = SOLVER_MAP_CONIC[solver]
        if (all(c in solver_instance.SUPPORTED_CONSTRAINTS for c in cones)
                and (has_constr or not solver_instance.REQUIRES_CONSTR)):
            reductions += [Dcp2Cone(),
                           CvxAttr2Constr(), ConeMatrixStuffing(),
                           solver_instance]
            return SolvingChain(reductions=reductions)

    raise SolverError("Either candidate conic solvers (%s) do not support the "
                      "cones output by the problem (%s), or there are not "
                      "enough constraints in the problem." % (
                          candidate_conic_solvers,
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

    def __init__(self, problem=None, reductions=[]):
        super(SolvingChain, self).__init__(problem=problem, reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError("Solving chains must terminate with a Solver.")
        self.solver = self.reductions[-1]

    def solve(self, problem, warm_start, verbose, solver_opts):
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

    def solve_via_data(self, problem, data, warm_start, verbose, solver_opts):
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
