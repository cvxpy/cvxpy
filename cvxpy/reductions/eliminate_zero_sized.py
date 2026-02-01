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


def replace_zero_sized(expr):
    """Recursively replace zero-sized expressions with zero constants."""
    if expr.size == 0:
        return Constant(np.zeros(expr.shape))
    if not expr.args:
        return expr
    new_args = []
    changed = False
    for arg in expr.args:
        new_arg = replace_zero_sized(arg)
        if new_arg is not arg:
            changed = True
        new_args.append(new_arg)
    if not changed:
        return expr
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

        # Walk each constraint's expression tree, replacing zero-sized
        # sub-expressions with zero constants and dropping zero-sized
        # constraints.
        constraints = []
        dropped_constraints = {}
        cons_id_map = {}
        any_changed = False
        for c in problem.constraints:
            if c.size == 0:
                any_changed = True
                dropped_constraints[c.id] = c
                continue
            new_args = []
            c_changed = False
            for arg in c.args:
                new_arg = replace_zero_sized(arg)
                if new_arg is not arg:
                    c_changed = True
                new_args.append(new_arg)
            if c_changed:
                any_changed = True
                data = c.get_data()
                if data is not None:
                    new_c = type(c)(*(new_args + data))
                else:
                    new_c = type(c)(*new_args)
                constraints.append(new_c)
                cons_id_map[c.id] = new_c.id
            else:
                constraints.append(c)

        # Replace zero-sized expressions in the objective.
        obj_expr = replace_zero_sized(problem.objective.expr)
        if obj_expr is not problem.objective.expr:
            any_changed = True
            objective = type(problem.objective)(obj_expr)
        else:
            objective = problem.objective

        if not any_changed and not zero_vars:
            return problem, {}

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
        inverse_data = {
            'eliminated_vars': eliminated_vars,
            'dropped_constraints': dropped_constraints,
            'cons_id_map': cons_id_map,
        }
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        if not inverse_data:
            return solution
        eliminated_vars = inverse_data.get('eliminated_vars', {})
        dropped_constraints = inverse_data.get('dropped_constraints', {})
        cons_id_map = inverse_data.get('cons_id_map', {})
        # Add default values for all eliminated variables.
        if solution.primal_vars is not None:
            for vid, var in eliminated_vars.items():
                solution.primal_vars[vid] = np.zeros(var.shape)
        # Map dual values back to original constraint IDs and add
        # zero duals for dropped zero-sized constraints.
        if solution.dual_vars is not None:
            for orig_id, new_id in cons_id_map.items():
                if new_id in solution.dual_vars:
                    solution.dual_vars[orig_id] = solution.dual_vars.pop(new_id)
            for cid, c in dropped_constraints.items():
                solution.dual_vars[cid] = np.zeros(c.shape)
        return solution
