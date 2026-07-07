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
from cvxpy import problems
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.reductions.reduction import Reduction
from cvxpy.utilities import scopes


def _fold_expr(expr):
    """Replace maximal non-affine variable-free parametric subtrees.

    A subtree with parameters but no variables that is NOT affine in the
    parameters (``power(t, 2)`` from DQCP bisection, ``floor(t)``, raw
    ``log_det(P)``) becomes a ``CallbackParam`` evaluating the subtree on
    each access: its value refreshes between solves, so downstream programs
    stay cacheable, and no canonicalizer ever sees it (a graph
    implementation would be unsound — the epigraph direction assumes
    params-affine curvature, so ``x <= power(t, 2)`` would relax vacuously).

    Parameter-affine subtrees (a bare ``Parameter``, ``2 * p + A``) are left
    intact: canonicalization and the backends handle them symbolically.
    """
    if not expr.parameters():
        return expr
    if not expr.variables():
        with scopes.dpp_scope():
            if expr.is_affine():
                return expr
        return CallbackParam(
            callback=lambda e=expr: e.value,
            shape=expr.shape,
            nonneg=expr.is_nonneg(),
            nonpos=expr.is_nonpos(),
        )
    new_args = [_fold_expr(arg) for arg in expr.args]
    if all(id(new) == id(old) for new, old in zip(new_args, expr.args)):
        return expr
    return expr.copy(new_args)


class CallbackParamFold(Reduction):
    """Fold non-affine parametric-constant subtrees into CallbackParams.

    Used on the ignore_dpp / non-DPP path (and the NLP path), where
    parameters stay symbolic across solves: unlike ``EvalParams`` this keeps
    the folded values refreshable, so the compiled program can be cached.
    """

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        if len(problem.objective.parameters()) > 0:
            obj_expr = _fold_expr(problem.objective.expr)
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective

        constraints = []
        for c in problem.constraints:
            args = [_fold_expr(arg) for arg in c.args]
            if all(id(new) == id(old) for new, old in zip(args, c.args)):
                constraints.append(c)
            else:
                data = c.get_data()
                if data is not None:
                    constraints.append(type(c)(*(args + data)))
                else:
                    constraints.append(type(c)(*args))
        return problems.problem.Problem(objective, constraints), []

    def invert(self, solution, inverse_data):
        return solution
