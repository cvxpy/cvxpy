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
from collections import namedtuple
from typing import Any, List, Tuple

import numpy as np

from cvxpy import problems
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.max import max as max_atom
from cvxpy.atoms.min import min as min_atom
from cvxpy.constraints import Inequality
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
from cvxpy.reductions.dqcp2dcp import inverse, sets, tighten
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution

# A tuple (feas_problem, param, tighten_lower, tighten_upper), where
#
#   `feas_problem` is a problem that can be used to check if the original
#       problem was feasible
#   `param` is the Parameter on which to bisect,
#   `tighten_lower` is a callable that takes a value of param for which
#       the problem is infeasible, and returns a larger value
#       for which the problem is still infeasible (or the smallest value for
#       which it is feasible)
#   `tighten_upper` is a callable that takes a value of param for which
#       the problem is feasible, and returns a smaller value for which the
#       problem is still feasible
BisectionData = namedtuple(
    "BisectionData",
    ['feas_problem', 'param', 'tighten_lower', 'tighten_upper'])


def _get_lazy_and_real_constraints(constraints):
    lazy_constraints = []
    real_constraints = []
    for c in constraints:
        if callable(c):
            lazy_constraints.append(c)
        else:
            real_constraints.append(c)
    return lazy_constraints, real_constraints


class Dqcp2Dcp(Canonicalization):
    """Reduce DQCP problems to a parameterized DCP problem.

    This reduction takes as input a DQCP problem and returns a parameterized
    DCP problem that can be solved by bisection. Some of the constraints might
    be lazy, i.e., callables that return a constraint when called. The problem
    will only be DCP once the lazy constraints are replaced with actual
    constraints.

    Problems emitted by this reduction can be solved with the `cp.bisect`
    function.
   """
    def __init__(self, problem=None) -> None:
        super(Dqcp2Dcp, self).__init__(
            canon_methods=CANON_METHODS, problem=problem)
        self._bisection_data = None

    def accepts(self, problem):
        """A problem is accepted if it is (a minimization) DQCP.
        """
        return type(problem.objective) == Minimize and problem.is_dqcp()

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        for vid in inverse_data.id_map:
            if vid not in pvars:
                # Variable was optimized out because it was unconstrained.
                pvars[vid] = 0.0
        return Solution(solution.status, solution.opt_val, pvars, {},
                        solution.attr)

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint.
        """
        constraints = []
        for constr in problem.constraints:
            constraints += self._canonicalize_constraint(constr)
        lazy, real = _get_lazy_and_real_constraints(constraints)
        feas_problem = problems.problem.Problem(Minimize(0), real)
        feas_problem._lazy_constraints = lazy

        objective = problem.objective.expr
        if objective.is_nonneg():
            t = Parameter(nonneg=True)
        elif objective.is_nonpos():
            t = Parameter(nonpos=True)
        else:
            t = Parameter()
        constraints += self._canonicalize_constraint(objective <= t)

        lazy, real = _get_lazy_and_real_constraints(constraints)
        param_problem = problems.problem.Problem(Minimize(0), real)
        param_problem._lazy_constraints = lazy
        param_problem._bisection_data = BisectionData(
            feas_problem, t, *tighten.tighten_fns(objective))
        return param_problem, InverseData(problem)

    def _canonicalize_tree(self, expr):
        canon_args, constrs = self._canon_args(expr)
        canon_expr, c = self.canonicalize_expr(expr, canon_args)
        constrs += c
        return canon_expr, constrs

    def _canon_args(self, expr) -> Tuple[List[Any], List[Any]]:
        """Canonicalize arguments of an expression.

        Like Canonicalization.canonicalize_tree, but preserves signs.
        """
        canon_args = []
        constrs = []
        for arg in expr.args:
            canon_arg, c = self._canonicalize_tree(arg)
            if isinstance(canon_arg, Variable):
                if arg.is_nonneg():
                    canon_arg.attributes["nonneg"] = True
                elif arg.is_nonpos():
                    canon_arg.attributes["nonpos"] = True
            canon_args += [canon_arg]
            constrs += c
        return canon_args, constrs

    def _canonicalize_constraint(self, constr):
        """Recursively canonicalize a constraint.

        The DQCP grammar has expresions of the form

            INCR* QCVX DCP

        and

            DECR* QCCV DCP

        ie, zero or more real/scalar increasing (or decreasing) atoms, composed
        with a quasiconvex (or quasiconcave) atom, composed with DCP
        expressions.

        The monotone functions are inverted by applying their inverses to
        both sides of a constraint. The QCVX (QCCV) atom is lowered by
        replacing it with its sublevel (superlevel) set. The DCP
        expressions are canonicalized via graph implementations.
        """
        lhs = constr.args[0]
        rhs = constr.args[1]

        if isinstance(constr, Inequality):
            # taking inverses can yield +/- infinity; this is handled here.
            lhs_val = np.array(lhs.value)
            rhs_val = np.array(rhs.value)
            if np.all(lhs_val == -np.inf) or np.all(rhs_val == np.inf):
                # constraint is redundant
                return [True]
            elif np.any(lhs_val == np.inf) or np.any(rhs_val == -np.inf):
                # constraint is infeasible
                return [False]

        if constr.is_dcp():
            canon_constr, aux_constr = self.canonicalize_tree(constr)
            return [canon_constr] + aux_constr

        # canonicalize lhs <= rhs
        # either lhs or rhs is quasiconvex (and not convex)
        assert isinstance(constr, Inequality)

        # short-circuit zero-valued expressions to simplify inverse logic
        if lhs.is_zero():
            return self._canonicalize_constraint(0 <= rhs)
        if rhs.is_zero():
            return self._canonicalize_constraint(lhs <= 0)

        if lhs.is_quasiconvex() and not lhs.is_convex():
            # quasiconvex <= constant
            assert rhs.is_constant(), rhs
            if inverse.invertible(lhs):
                # Apply inverse to both sides of constraint.
                rhs = inverse.inverse(lhs)(rhs)
                idx = lhs._non_const_idx()[0]
                expr = lhs.args[idx]
                if lhs.is_incr(idx):
                    return self._canonicalize_constraint(expr <= rhs)
                assert lhs.is_decr(idx)
                return self._canonicalize_constraint(expr >= rhs)
            elif isinstance(lhs, (maximum, max_atom)):
                # Lower maximum.
                return [c for arg in lhs.args
                        for c in self._canonicalize_constraint(arg <= rhs)]
            else:
                # Replace quasiconvex atom with a sublevel set.
                canon_args, aux_args_constr = self._canon_args(lhs)
                sublevel_set = sets.sublevel(lhs.copy(canon_args), t=rhs)
                return sublevel_set + aux_args_constr

        # constant <= quasiconcave
        assert rhs.is_quasiconcave()
        assert lhs.is_constant()
        if inverse.invertible(rhs):
            # Apply inverse to both sides of constraint.
            lhs = inverse.inverse(rhs)(lhs)
            idx = rhs._non_const_idx()[0]
            expr = rhs.args[idx]
            if rhs.is_incr(idx):
                return self._canonicalize_constraint(lhs <= expr)
            assert rhs.is_decr(idx)
            return self._canonicalize_constraint(lhs >= expr)
        elif isinstance(rhs, (minimum, min_atom)):
            # Lower minimum.
            return [c for arg in rhs.args
                    for c in self._canonicalize_constraint(lhs <= arg)]
        else:
            # Replace quasiconcave atom with a superlevel set.
            canon_args, aux_args_constr = self._canon_args(rhs)
            superlevel_set = sets.superlevel(rhs.copy(canon_args), t=lhs)
            return superlevel_set + aux_args_constr
