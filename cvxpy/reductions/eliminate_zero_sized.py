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

import numpy as np

from cvxpy.expressions.constants.constant import Constant
from cvxpy.reductions.reduction import Reduction


def _has_zero_sized_var(expr, zero_var_ids):
    """Check if any variable in expr is zero-sized."""
    for v in expr.variables():
        if v.id in zero_var_ids:
            return True
    return False


def replace_zero_sized(expr, zero_var_ids):
    """Recursively replace zero-sized expressions with zero constants."""
    if expr.size == 0:
        return Constant(np.zeros(expr.shape))
    if not _has_zero_sized_var(expr, zero_var_ids):
        return expr
    new_args = [replace_zero_sized(arg, zero_var_ids) for arg in expr.args]
    return expr.copy(new_args)


class EliminateZeroSized(Reduction):
    """Eliminates zero-sized expressions from a problem.

    Zero-sized variables (those with a 0 in their shape) have no elements
    and are vacuous. This reduction:
    - Drops constraints that are entirely zero-sized (vacuously true).
    - Replaces zero-sized variables with zero constants so downstream
      reductions never see them.
    - Records eliminated variables so their values can be filled in
      during solution inversion.
    """

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        from cvxpy.problems.problem import Problem

        zero_vars = {v.id: v for v in problem.variables() if v.size == 0}

        # Filter out zero-sized constraints and replace zero-sized vars.
        constraints = []
        any_changed = bool(zero_vars)
        for c in problem.constraints:
            if c.size == 0:
                any_changed = True
                continue
            if zero_vars and _has_zero_sized_var(c, zero_vars):
                new_args = [replace_zero_sized(arg, zero_vars)
                            for arg in c.args]
                data = c.get_data()
                if data is not None:
                    constraints.append(type(c)(*(new_args + data)))
                else:
                    constraints.append(type(c)(*new_args))
            else:
                constraints.append(c)

        if not any_changed:
            return problem, {}

        # Replace zero-sized expressions in the objective.
        if zero_vars:
            obj_expr = replace_zero_sized(problem.objective.expr, zero_vars)
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective

        # Find variables that were in the original problem but are
        # absent from the reduced problem.
        reduced_problem = Problem(objective, constraints)
        new_var_ids = {v.id for v in reduced_problem.variables()}
        eliminated_vars = {}
        for v in problem.variables():
            if v.id in new_var_ids:
                continue
            if v.size > 0 and v.domain:
                # Non-zero-sized variables with domain constraints
                # (bounds, nonneg, etc.) need those constraints
                # enforced. Add them back so downstream reductions
                # (and solvers that handle bounds natively) see them.
                constraints = constraints + v.domain
            else:
                # Zero-sized variables and unconstrained orphaned
                # variables are truly eliminated.
                eliminated_vars[v.id] = v

        new_problem = Problem(objective, constraints)
        return new_problem, eliminated_vars

    def invert(self, solution, inverse_data):
        eliminated_vars = inverse_data
        if not eliminated_vars:
            return solution
        # Add default values for all eliminated variables.
        if solution.primal_vars is not None:
            for vid, var in eliminated_vars.items():
                solution.primal_vars[vid] = np.zeros(var.shape)
        return solution
