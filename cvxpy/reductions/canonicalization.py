"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy import problems
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants import Constant
from cvxpy.reductions import InverseData, Reduction, Solution
from cvxpy.expressions.constants import CallbackParam


class Canonicalization(Reduction):
    """TODO(akshayka): Document this class."""

    def __init__(self, canon_methods=None):
        self.canon_methods = canon_methods

    def apply(self, problem):
        inverse_data = InverseData(problem)

        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective)

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            canon_constr, aux_constr = self.canonicalize_tree(
                constraint)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id:
                                             canon_constr.id})

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

    def canonicalize_tree(self, expr):
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            canon_expr, constrs = self.canonicalize_tree(
              expr.args[0].objective.expr)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(constr)
                constrs += [canon_constr] + aux_constr
        else:
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(arg)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args)
            constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args):
        if isinstance(expr, Expression) and not expr.variables():
            # Parameterized expressions are evaluated in a subsequent
            # reduction.
            if expr.parameters():
                rows, cols = expr.shape
                param = CallbackParam(lambda: expr.value, (rows, cols))
                return param, []
            # Non-parameterized expressions are evaluated immediately.
            else:
                return Constant(expr.value), []
        elif type(expr) in self.canon_methods:
            return self.canon_methods[type(expr)](expr, args)
        else:
            return expr.copy(args), []
