from cvxpy.error import DCPError, DGPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions import (Chain, Dcp2Cone,
                              FlipObjective, Dgp2Dcp, Qp2SymbolicQp,
                              CvxAttr2Constr, Complex2Real)
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.solvers import defines as slv_def


def construct_symbolic_chain(problem, solver=None, gp=False):
    """Build a symbolic chain from a problem to its symbolic form.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.

    Returns
    -------
    Chain
        A Chain that can be used to convert the problem to symbolic form.

    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True.
    """
    if solver is not None:
        if solver not in slv_def.INSTALLED_SOLVERS:
            raise SolverError("The solver %s is not installed." % solver)
        candidates = [solver]
    else:
        candidates = slv_def.INSTALLED_SOLVERS

    if gp:
        if solver is not None and solver not in slv_def.CONIC_SOLVERS:
            raise SolverError(
              "When `gp=True`, `solver` must be a conic solver "
              "(received '%s'); try calling `solve()` with `solver=cvxpy.ECOS`."
              % solver)
        elif solver is None:
            candidates = slv_def.INSTALLED_CONIC_SOLVERS

    reductions = []
    if len(problem.variables()) == 0:
        return Chain(reductions=reductions)
    if complex2real.accepts(problem):
        reductions += [Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]

    if not gp and not problem.is_dcp():
        append = ""
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
        reductions += [FlipObjective()]

    # Conclude the chain with either a symbolic QP or a symbolic conic program

    # First, attempt to canonicalize the problem to a linearly constrained QP.
    candidate_qp_solvers = [s for s in slv_def.QP_SOLVERS if s in candidates]
    # Consider only MIQP solvers if problem is integer
    if problem.is_mixed_integer():
        candidate_qp_solvers = [
          s for s in candidate_qp_solvers
          if slv_def.SOLVER_MAP_QP[s].MIP_CAPABLE]
    if candidate_qp_solvers and qp2symbolic_qp.accepts(problem):
        reductions += [CvxAttr2Constr(),
                       Qp2SymbolicQp()]
        return Chain(reductions=reductions)

    # Canonicalize to conic otherwise
    candidate_conic_solvers = [s for s in slv_def.CONIC_SOLVERS
                               if s in candidates]
    if problem.is_mixed_integer():
        candidate_conic_solvers = \
            [s for s in candidate_conic_solvers if
             slv_def.SOLVER_MAP_CONIC[s].MIP_CAPABLE]
        if not candidate_conic_solvers and \
                not candidate_qp_solvers:
            raise SolverError("Problem is mixed-integer, but candidate "
                              "QP/Conic solvers (%s) are not MIP-capable." %
                              [candidate_qp_solvers, candidate_conic_solvers])
    if not candidate_conic_solvers:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)
    reductions += [Dcp2Cone(),
                   CvxAttr2Constr()]
    return Chain(reductions=reductions)

