from cvxpy.atoms import EXP_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import ExpCone, PSD, SOC
from cvxpy.error import SolverError
from cvxpy.reductions import (Chain, ConeMatrixStuffing, EvalParams,
                              QpMatrixStuffing)
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers import defines as slv_def


def construct_solving_chain(problem, candidates):
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
    reductions = []
    if problem.parameters():
        reductions += [EvalParams()]
    if len(problem.variables()) == 0:
        reductions += [ConstantSolver()]
        return SolvingChain(reductions=reductions)

    # Conclude the chain with one of the following:
    #   (1) QpMatrixStuffing --> [a QpSolver],
    #   (2) ConeMatrixStuffing --> [a ConicSolver]

    # First, attempt to canonicalize the problem to a linearly constrained QP.
    if candidates['qp_solvers'] and QpMatrixStuffing.accepts(problem):
        solver = sorted(candidates['qp_solvers'],
                        key=lambda s: slv_def.QP_SOLVERS.index(s))[0]
        solver_instance = slv_def.SOLVER_MAP_QP[solver]
        reductions += [QpMatrixStuffing(),
                       solver_instance]
        return SolvingChain(reductions=reductions)

    if not candidates['conic_solvers']:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)

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

    for solver in sorted(candidates['conic_solvers'],
                         key=lambda s: slv_def.CONIC_SOLVERS.index(s)):
        solver_instance = slv_def.SOLVER_MAP_CONIC[solver]
        if (all(c in solver_instance.SUPPORTED_CONSTRAINTS for c in cones)
                and (has_constr or not solver_instance.REQUIRES_CONSTR)):
            reductions += [ConeMatrixStuffing(),
                           solver_instance]
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

    def __init__(self, problem=None, reductions=[]):
        super(SolvingChain, self).__init__(problem=problem,
                                           reductions=reductions)
        if not isinstance(self.reductions[-1], Solver):
            raise ValueError("Solving chains must terminate with a Solver.")
        self.solver = self.reductions[-1]

    def prepend(self, chain):
        """
        Create and return a new SolvingChain by concatenating
        chain with this instance.
        """
        return SolvingChain(reductions=chain.reductions + self.reductions)

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

    def solve_via_data(self, problem, data, warm_start=False, verbose=False,
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
