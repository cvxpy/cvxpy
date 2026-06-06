"""
Copyright 2025 CVXPY developers

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
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dnlp2smooth.canonicalizers import SMOOTH_CANON_METHODS as smooth_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.subexpr_cache import (
    ExprKey,
    StructuralKeyCache,
    UncacheableError,
    expr_key,
)


class Dnlp2Smooth(Canonicalization):
    """
    Reduce a disciplined nonlinear program to an equivalent smooth program

    This reduction takes as input (minimization) expressions and converts
    them into smooth expressions.
    """
    def __init__(self, problem=None) -> None:
        super(Canonicalization, self).__init__(problem=problem)
        self.smooth_canon_methods = smooth_canon_methods

    def accepts(self, problem):
        """A problem is always accepted"""
        return True

    def apply(self, problem):
        """Converts an expr to a smooth program"""
        inverse_data = InverseData(problem)

        inverse_data.minimize = type(problem.objective) == Minimize

        cse_cache = {}
        structural_key_cache = StructuralKeyCache()

        # smoothen objective function
        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective, True, cse_cache, structural_key_cache)

        # smoothen constraints
        for constraint in problem.constraints:
            # canon_constr is the constraint re-expressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            canon_constr, aux_constr = self.canonicalize_tree(
                constraint, False, cse_cache, structural_key_cache)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map[constraint.id] = canon_constr.id

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        self._cons_id_map = inverse_data.cons_id_map
        return new_problem, inverse_data

    def canonicalize_tree(
        self,
        expr,
        affine_above: bool,
        cse_cache: dict[ExprKey, Expression] | None = None,
        structural_key_cache: StructuralKeyCache | None = None,
    ) -> tuple[Expression, list[Constraint]]:
        """Recursively canonicalize an Expression.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        if cse_cache is None:
            cse_cache = {}
        if structural_key_cache is None:
            structural_key_cache = StructuralKeyCache()

        # Only Expression subtrees are eligible for the CSE cache. Objectives
        # and user constraints are intentionally excluded so their IDs flow
        # through to inverse_data unchanged. partial_problem subtrees carry
        # their own constraints with IDs and are excluded for the same reason.
        partial_problem_cls = cvxtypes.partial_problem()
        cache_eligible = (
            isinstance(expr, Expression) and type(expr) != partial_problem_cls
        )
        cache_key = None
        if cache_eligible:
            try:
                cache_key = expr_key(expr, structural_key_cache)
            except UncacheableError:
                cache_key = None
            if cache_key is not None and cache_key in cse_cache:
                return cse_cache[cache_key], []

        if type(expr) == partial_problem_cls:
            canon_expr, constrs = self.canonicalize_tree(
                expr.args[0].objective.expr, False, cse_cache,
                structural_key_cache)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(
                    constr, False, cse_cache, structural_key_cache)
                constrs += [canon_constr] + aux_constr
        else:
            affine_atom = type(expr) not in self.smooth_canon_methods
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(
                    arg, affine_atom and affine_above,
                    cse_cache, structural_key_cache)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args, affine_above)
            constrs += c

        if cache_key is not None:
            cse_cache[cache_key] = canon_expr
        return canon_expr, constrs

    def canonicalize_expr(
        self,
        expr: Expression,
        args: list[Expression],
        affine_above: bool,
    ) -> tuple[Expression, list[Constraint]]:
        """Canonicalize an expression, w.r.t. canonicalized arguments.

        Parameters
        ----------
        expr : The expression tree to canonicalize.
        args : The canonicalized arguments of expr.
        affine_above : The path up to the root node is all affine atoms.

        Returns
        -------
        A tuple of the canonicalized expression and generated constraints.
        """
        # Constant trees are collapsed, but parameter trees are preserved.
        if isinstance(expr, Expression) and (
                expr.is_constant() and not expr.parameters()):
            return expr, []

        if type(expr) in self.smooth_canon_methods:
            return self.smooth_canon_methods[type(expr)](expr, args)

        return expr.copy(args), []
