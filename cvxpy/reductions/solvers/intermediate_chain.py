from cvxpy.atoms.atom import Atom
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

    def build_dcp_error_msg():
        def find_non_dcp_leaves(expr, res=[]):
            def is_dcp(e):
                if isinstance(e, Atom):
                    return e.is_convex()
                return e.is_dcp()

            if (len(expr.args) == 0 and expr.is_dcp()):
                return res
            if (not is_dcp(expr)) and all(is_dcp(child) for child in expr.args):
                res.append(expr)
            for child in expr.args:
                res = find_non_dcp_leaves(child, res)
            return res

        if not problem.objective.is_dcp():
            non_dcp_leaves = find_non_dcp_leaves(problem.objective.expr)
            msg = "The objective is not DCP. Its following subexpressions are not:"
            for expr in non_dcp_leaves:
                msg += '\n%s' % (str(expr,))
            return msg
        not_dcp_constraints = [expr for expr in problem.constraints if not expr.is_dcp()]
        msg = "The following constraints are not DCP:"
        for expr in not_dcp_constraints:
            msg += '\n%s , because the following subexpressions are not:' % (expr,)
            non_dcp_leaves = find_non_dcp_leaves(expr)
            for subexpr in non_dcp_leaves:
                msg += '\n|--  %s' % (str(subexpr,))
        return msg

    if not gp and not problem.is_dcp():
        append = build_dcp_error_msg()
        if problem.is_dgp():
            append = ("\nHowever, the problem does follow DGP rules. "
                      "Consider calling this function with `gp=True`.")
        raise DCPError("Problem does not follow DCP rules. Specifically:\n" + append)
    elif gp and not problem.is_dgp():
        append = ""
        if problem.is_dcp():
            append = (" However, the problem does follow DCP rules. "
                      "Consider calling this function with `gp=False`.")
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
