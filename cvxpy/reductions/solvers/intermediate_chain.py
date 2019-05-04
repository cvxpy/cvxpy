from cvxpy.error import DCPError, DGPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions import (Chain, Dcp2Cone,
                              FlipObjective, Dgp2Dcp, Qp2SymbolicQp,
                              CvxAttr2Constr, Complex2Real)
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp


def construct_intermediate_chain(problem, candidates, gp=False):
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
    Chain
        A Chain that can be used to convert the problem to an intermediate form.

    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True.
    """

    reductions = []
    if len(problem.variables()) == 0:
        return Chain(reductions=reductions)
    # TODO Handle boolean constraints.
    if complex2real.accepts(problem):
        reductions += [Complex2Real()]
    if gp:
        reductions += [Dgp2Dcp()]

    if not gp and not problem.is_dcp():
        append = ""
        if problem.is_dgp():
            append = (" However, the problem does follow DGP rules. "
                      "Consider calling solve() with `gp=True`.")
        elif problem.is_dqcp():
            append = (" However, the problem does follow DQCP rules. "
                      "Consider calling solve() with `qcp=True`.")
        raise DCPError("Problem does not follow DCP rules." + append)
    elif gp and not problem.is_dgp():
        append = ""
        if problem.is_dcp():
            append = (" However, the problem does follow DCP rules. "
                      "Consider calling solve() with `gp=False`.")
        elif problem.is_dqcp():
            append = (" However, the problem does follow DQCP rules. "
                      "Consider calling solve() with `qcp=True`.")
        raise DGPError("Problem does not follow DGP rules." + append)

    # Dcp2Cone and Qp2SymbolicQp require problems to minimize their objectives.
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]

    # First, attempt to canonicalize the problem to a linearly constrained QP.
    if candidates['qp_solvers'] and qp2symbolic_qp.accepts(problem):
        reductions += [CvxAttr2Constr(),
                       Qp2SymbolicQp()]
        return Chain(reductions=reductions)

    # Canonicalize it to conic problem.
    if not candidates['conic_solvers']:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)
    reductions += [Dcp2Cone(),
                   CvxAttr2Constr()]
    return Chain(reductions=reductions)
