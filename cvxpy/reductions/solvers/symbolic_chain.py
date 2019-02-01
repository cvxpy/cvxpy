from cvxpy.atoms import EXP_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import ExpCone, PSD, SOC
from cvxpy.error import DCPError, DGPError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions import (Chain, Dcp2Cone,
                              FlipObjective, Dgp2Dcp, Qp2SymbolicQp,
                              CvxAttr2Constr, Complex2Real)
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp


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

    reductions = []
    #  if problem.parameters():
    #      reductions += [EvalParams()]
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
        reductions.append(FlipObjective())

    # Conclude the chain with either a symbolic QP or a symbolic conic program

    # First, attempt to canonicalize the problem to a linearly constrained QP.
    if qp2symbolic_qp.accepts(problem):
        reductions += [CvxAttr2Constr(),
                       Qp2SymbolicQp()]
        return Chain(reductions=reductions)

    # Canonicalize to conic otherwise
    reductions += [Dcp2Cone(),
                   CvxAttr2Constr()]
    return Chain(reductions=reductions)

