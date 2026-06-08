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


from typing import overload

from cvxpy import problems
from cvxpy.atoms.elementwise.power import Power
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize, Objective
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS as cone_canon_methods
from cvxpy.reductions.dcp2cone.canonicalizers.quad import QUAD_CANON_METHODS as quad_canon_methods
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.subexpr_cache import (
    ExprKey,
    StructuralKeyCache,
    UncacheableError,
    expr_key,
)
from cvxpy.utilities.canonical import Canonical

# A Dcp2Cone CSE cache key: a subtree's structural key, paired with
# affine_above only when canonicalization could depend on it (the quad-objective
# branch); None otherwise so cone-mode subtrees merge across contexts.
ConeCacheKey = tuple[ExprKey, bool | None]


class Dcp2Cone(Canonicalization):
    """Reduce DCP problems to a conic form.

    This reduction takes as input (minimization) DCP problems and converts
    them into problems with affine or quadratic objectives and conic
    constraints whose arguments are affine.
    """
    def __init__(self, problem=None, quad_obj: bool = False, solver_context=None) -> None:
        super(Canonicalization, self).__init__(problem=problem)
        self.cone_canon_methods = cone_canon_methods
        self.quad_canon_methods = quad_canon_methods
        self.quad_obj = quad_obj

        # solver_context : The solver context: supported constraints and bounds.
        self.solver_context = solver_context

    def accepts(self, problem):
        """A problem is accepted if it is a minimization and is DCP.
        """
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        """Converts a DCP problem to a conic form.
        """
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to cone program")

        inverse_data = InverseData(problem)

        cse_cache: dict[ConeCacheKey, Expression] = {}
        structural_key_cache = StructuralKeyCache()
        affine_above_relevant_cache: dict[int, bool] = {}

        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective, True, cse_cache,
            structural_key_cache, affine_above_relevant_cache)

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            canon_constr, aux_constr = self.canonicalize_tree(
                constraint, False, cse_cache,
                structural_key_cache, affine_above_relevant_cache)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map[constraint.id] = canon_constr.id

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        self._cons_id_map = inverse_data.cons_id_map
        return new_problem, inverse_data

    @overload
    def canonicalize_tree(
        self,
        expr: Expression,
        affine_above: bool,
        cse_cache: dict[ConeCacheKey, Expression] | None = None,
        structural_key_cache: StructuralKeyCache | None = None,
        affine_above_relevant_cache: dict[int, bool] | None = None,
    ) -> tuple[Expression, list[Constraint]]: ...
    @overload
    def canonicalize_tree(
        self,
        expr: Constraint,
        affine_above: bool,
        cse_cache: dict[ConeCacheKey, Expression] | None = None,
        structural_key_cache: StructuralKeyCache | None = None,
        affine_above_relevant_cache: dict[int, bool] | None = None,
    ) -> tuple[Constraint, list[Constraint]]: ...
    @overload
    def canonicalize_tree(
        self,
        expr: Objective,
        affine_above: bool,
        cse_cache: dict[ConeCacheKey, Expression] | None = None,
        structural_key_cache: StructuralKeyCache | None = None,
        affine_above_relevant_cache: dict[int, bool] | None = None,
    ) -> tuple[Objective, list[Constraint]]: ...

    def canonicalize_tree(
        self,
        expr: Canonical,
        affine_above: bool,
        cse_cache: dict[ConeCacheKey, Expression] | None = None,
        structural_key_cache: StructuralKeyCache | None = None,
        affine_above_relevant_cache: dict[int, bool] | None = None,
    ) -> tuple[Canonical, list[Constraint]]:
        """Recursively canonicalize an Expression.

        Canonicalizing an Expression yields an Expression, a Constraint yields
        a Constraint, and an Objective yields an Objective; the overloads above
        preserve that distinction for callers.

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
        if affine_above_relevant_cache is None:
            affine_above_relevant_cache = {}

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
            cache_key = self._make_cache_key(
                expr, affine_above, structural_key_cache,
                affine_above_relevant_cache)
            if cache_key is not None and cache_key in cse_cache:
                return cse_cache[cache_key], []

        # TODO don't copy affine expressions?
        if type(expr) == partial_problem_cls:
            canon_expr, constrs = self.canonicalize_tree(
                expr.args[0].objective.expr, False, cse_cache,
                structural_key_cache, affine_above_relevant_cache)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(
                    constr, False, cse_cache,
                    structural_key_cache, affine_above_relevant_cache)
                constrs += [canon_constr] + aux_constr
        else:
            affine_atom = type(expr) not in self.cone_canon_methods
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(
                    arg, affine_atom and affine_above, cse_cache,
                    structural_key_cache, affine_above_relevant_cache)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args, affine_above)
            constrs += c

        if cache_key is not None:
            # Only canon_expr is needed on hit; subsequent uses return an
            # empty constraint list because the first emission already
            # added the generated constraints to the caller's list.
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

        if self.quad_obj and affine_above and type(expr) in self.quad_canon_methods:
            # Special case for power.
            # isinstance catches both Power and PowerApprox
            if isinstance(expr, Power) and not expr._quadratic_power():
                return self.cone_canon_methods[type(expr)](expr, args,
                                                           solver_context=self.solver_context)
            # quad_over_lin requires a constant denominator for the quad canon.
            # Check the canonicalized args (not expr.args) since children may
            # have been replaced with auxiliary variables.
            elif type(expr) == quad_over_lin and not args[1].is_constant():
                return self.cone_canon_methods[type(expr)](expr, args,
                                                           solver_context=self.solver_context)
            else:
                return self.quad_canon_methods[type(expr)](expr, args,
                                                           solver_context=self.solver_context)

        if type(expr) in self.cone_canon_methods:
            return self.cone_canon_methods[type(expr)](expr, args,
                                                       solver_context=self.solver_context)

        return expr.copy(args), []

    # ----- CSE cache key construction -----

    def _make_cache_key(
        self,
        expr: Expression,
        affine_above: bool,
        structural_key_cache: StructuralKeyCache,
        affine_above_relevant_cache: dict[int, bool],
    ) -> ConeCacheKey | None:
        """Build a hashable structural key for an Expression subtree.

        Returns None if a safe key cannot be built, in which case the caller
        skips caching for that subtree rather than risking incorrect reuse.
        ``affine_above`` is included only when the result actually depends on
        it (i.e. quad-objective canonicalization could fire for this subtree),
        so cone-mode canonicalization cannot share a result with the quad
        branch but identical subtrees that happen to sit under affine-only and
        non-affine roots still merge.
        """
        try:
            structural = expr_key(expr, structural_key_cache)
        except UncacheableError:
            return None
        if self._affine_above_relevant(expr, affine_above_relevant_cache):
            return (structural, bool(affine_above))
        return (structural, None)

    def _affine_above_relevant(
        self, expr: Canonical, affine_above_relevant_cache: dict[int, bool]
    ) -> bool:
        """Whether canonicalize_tree result for ``expr`` depends on affine_above.

        Returns True when ``expr`` itself or any descendant could take the
        quad-canon path (which requires self.quad_obj and an unbroken chain of
        affine atoms from the root). For purely cone-mode subtrees, the result
        is independent of affine_above and caching can merge across contexts.
        """
        if not self.quad_obj or not isinstance(expr, Expression):
            return False
        cache_key = id(expr)
        if cache_key in affine_above_relevant_cache:
            return affine_above_relevant_cache[cache_key]
        if type(expr) in self.quad_canon_methods:
            relevant = True
        elif type(expr) in self.cone_canon_methods:
            # A non-affine atom forces affine_above=False for its children, so
            # descendants cannot reach the quad branch through this node.
            relevant = False
        else:
            # Affine atom: forwards affine_above to children, so check descendants.
            relevant = any(
                self._affine_above_relevant(arg, affine_above_relevant_cache)
                for arg in expr.args)
        affine_above_relevant_cache[cache_key] = relevant
        return relevant
