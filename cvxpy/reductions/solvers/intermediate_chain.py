"""
Copyright, the CVXPY authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.error import DCPError, DGPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg


def construct_intermediate_chain(problem, candidates, gp: bool = False):
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
        raise DCPError("Problem does not follow DCP rules. Specifically:\n" + append)

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
