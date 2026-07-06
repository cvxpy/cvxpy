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
from cvxpy.error import ParameterError
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.reductions.reduction import Reduction


def fold_variable_free_subtrees(expr):
    """Fold variable-free *composite* parametric subtrees to their values.

    Bare ``Parameter`` leaves are kept symbolic so that variable-coupled
    parameter usage (e.g. ``A @ x``, ``square(p * x)``) reaches the DIFFENGINE
    backend, which re-evaluates it on each solve. Only *composite* subtrees
    that are parametric and contain no optimization variables (e.g.
    ``log_det(P)``, ``A @ B``) are folded to constants -- these are the terms
    that would otherwise needlessly force cones or break the
    affine-in-parameter structure.
    """
    if isinstance(expr, list):
        return [fold_variable_free_subtrees(elem) for elem in expr]
    elif len(expr.parameters()) == 0:
        return expr
    elif isinstance(expr, Parameter):
        # Keep bare parameters symbolic.
        return expr
    elif len(expr.variables()) == 0:
        # Variable-free parametric composite: fold the whole subtree to its
        # numeric value (it only contributes a constant offset to the program).
        if expr.value is None:
            raise ParameterError("Problem contains unspecified parameters.")
        return Constant(expr.value)
    else:
        new_args = []
        for arg in expr.args:
            new_args.append(fold_variable_free_subtrees(arg))
        return expr.copy(new_args)


class FoldVariableFreeParams(Reduction):
    """Folds variable-free composite parametric subtrees to constants.

    The DIFFENGINE companion of :class:`~cvxpy.reductions.eval_params.EvalParams`:
    where ``EvalParams`` bakes every parameter, this reduction folds only the
    variable-free parametric *composites* (e.g. ``log_det(P)``, ``norm(p)``),
    which avoids needlessly forcing cones, and keeps bare parameters symbolic
    for the DIFFENGINE backend to re-evaluate on each solve. Because the
    folded constants embed current parameter values, a chain containing this
    reduction must not cache its parametric program (see ``safe_to_cache`` in
    ``cvxpy/problems/problem.py``).
    """

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        """Fold variable-free composite parametric subtrees.

        Raises
        ------
        ParameterError
            If a folded subtree contains a parameter whose value is None.
        """
        # Do not instantiate a new objective if it does not contain
        # parameters.
        if len(problem.objective.parameters()) > 0:
            obj_expr = fold_variable_free_subtrees(problem.objective.expr)
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective

        constraints = []
        for c in problem.constraints:
            args = [fold_variable_free_subtrees(arg) for arg in c.args]
            # Do not instantiate a new constraint object if it did not
            # contain parameters.
            if all(id(new) == id(old) for new, old in zip(args, c.args)):
                constraints.append(c)
            # Otherwise, create a copy of the constraint.
            else:
                data = c.get_data()
                if data is not None:
                    constraints.append(type(c)(*(args + data)))
                else:
                    constraints.append(type(c)(*args))
        return problems.problem.Problem(objective, constraints), []

    def invert(self, solution, inverse_data):
        """Returns a solution to the original problem given the inverse_data.
        """
        return solution
