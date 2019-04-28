"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain
a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from cvxpy import problems
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.constraints import Inequality
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS
from cvxpy.reductions.dqcp2dcp import inverse
from cvxpy.reductions.dqcp2dcp import tighten
from cvxpy.reductions.dqcp2dcp import sets
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution

from collections import namedtuple


BisectionData = namedtuple(
    "BisectionData",
    ['problem', 'param', 'tighten_lower', 'tighten_upper'])


class Dqcp2Dcp(Canonicalization):
    """Reduce DQCP problems to a parameterized DCP problem.

    This reduction takes as input a DQCP problem and returns a parameterized DCP
    problem that can be solved by bisection.

    Example
    -------

    # TODO(akshayka): Convert this to a DQCP example.
    >>> import cvxpy as cp
    >>>
    >>> x1 = cp.Variable(pos=True)
    >>> x2 = cp.Variable(pos=True)
    >>> x3 = cp.Variable(pos=True)
    >>>
    >>> monomial = 3.0 * x_1**0.4 * x_2 ** 0.2 * x_3 ** -1.4
    >>> posynomial = monomial + 2.0 * x_1 * x_2
    >>> dgp_problem = cp.Problem(cp.Minimize(posynomial), [monomial == 4.0])
    >>>
    >>> dcp2cone = cvxpy.reductions.Dcp2Cone()
    >>> assert not dcp2cone.accepts(dgp_problem)
    >>>
    >>> gp2dcp = cvxpy.reductions.Dgp2Dcp(dgp_problem)
    >>> dcp_problem = gp2dcp.reduce()
    >>>
    >>> assert dcp2cone.accepts(dcp_problem)
    >>> dcp_probem.solve()
    >>>
    >>> dgp_problem.unpack(gp2dcp.retrieve(dcp_problem.solution))
    >>> print(dgp_problem.value)
    >>> print(dgp_problem.variables())
    """
    def __init__(self, problem=None):
        super(Dqcp2Dcp, self).__init__(
            canon_methods=CANON_METHODS, problem=problem)
        self._bisection_data = None

    def accepts(self, problem):
        """A problem is accepted if it is (a minimization) DQCP.
        """
        return type(problem.objective) == Minimize and problem.is_dqcp()

    def invert(self, solution, inverse_data):
        # Convex duality doesn't apply to quasiconvex problems.
        # TODO might not make sense, just need to take x.value, put it into
        # new prob
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        return Solution(solution.status, solution.opt_val, pvars, {},
                        solution.attr)

    @property
    def bisection_data(self):
        return self._bisection_data

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        objective = problem.objective.expr
        if objective.is_nonneg():
            t = Parameter(nonneg=True)
        elif objective.is_nonpos():
            t = Parameter(nonpos=True)
        else:
            t = Parameter()

        constraints = []
        for constr in [objective <= t] + problem.constraints:
            constraints += self._canonicalize_constraint(constr)
        param_problem = problems.problem.Problem(Minimize(0), constraints)
        self._bisection_data = BisectionData(
            param_problem, t, *tighten.tighten_fns(objective))
        return param_problem, InverseData(problem)

    def _canon_args(self, expr):
        canon_args = []
        constrs = []
        for arg in expr.args:
            canon_arg, c = self.canonicalize_tree(arg)
            canon_args += [canon_arg]
            constrs += c
        return canon_args, constrs

    def _canonicalize_constraint(self, constr):
        """Recursively canonicalize a constraint."""
        if constr.is_dcp():
            canon_constr, aux_constr = self.canonicalize_tree(constr)
            return [canon_constr] + aux_constr

        # canonicalize lhs <= rhs
        # either lhs or rhs is quasiconvex (and not convex)
        assert isinstance(constr, Inequality)
        lhs = constr.args[0]
        rhs = constr.args[1]
        if lhs.is_quasiconvex() and not lhs.is_convex():
            # canonicalize quasiconvex <= constant
            assert rhs.is_constant()
            if inverse.invertible(lhs):
                rhs = inverse.inverse(lhs)(rhs)
                if lhs.is_incr(0):
                    return self._canonicalize_constraint(lhs.args[0] <= rhs)
                return self._canonicalize_constraint(lhs.args[0] >= rhs)
            elif isinstance(lhs, maximum):
                return [c for arg in lhs.args
                        for c in self._canonicalize_constraint(arg <= rhs)]
            else:
                canon_args, aux_args_constr = self._canon_args(lhs)
                sublevel_set = sets.sublevel(lhs.copy(canon_args), t=rhs)
                return sublevel_set + aux_args_constr

        # canonicalize constant <= quasiconcave
        assert rhs.is_quasiconcave()
        assert lhs.is_constant()
        if inverse.invertible(rhs):
            lhs = inverse.inverse(rhs)(lhs)
            if rhs.is_incr(0):
                return self._canonicalize_constraint(lhs <= rhs.args[0])
            return self._canonicalize_constraint(lhs >= inverse(lhs)(rhs))
        elif isinstance(rhs, minimum):
            return [c for arg in rhs.args
                    for c in self._canonicalize_constraint(lhs <= arg)]
        else:
            canon_args, aux_args_constr = self._canon_args(rhs)
            superlevel_set = sets.superlevel(rhs.copy(canon_args), t=lhs)
            return superlevel_set + aux_args_constr
