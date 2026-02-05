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

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from cvxpy.atoms.atom import Atom
    from cvxpy.constraints.constraint import Constraint
    from cvxpy.utilities.solver_context import SolverInfo


def get_expr_value_if_supported(
    expr: Atom,
    solver_context: Optional[SolverInfo],
) -> Optional[np.ndarray]:
    """Compute the value of an expression if warm starting is supported.

    Returns None if:
    - solver_context is None
    - The solver does not support warm starting
    - The expression's value is None (e.g., variables have no value set)

    Otherwise, returns a dense ndarray matching expr.shape.
    """
    if solver_context is None or not solver_context.solver_supports_warm_start:
        return None
    val = expr.value
    if val is None:
        return None
    return np.atleast_1d(np.array(val, dtype=float)).reshape(expr.shape)


def propagate_dual_values_to_constraints(
    expr: Atom,
    new_constraints: List[Constraint],
    solver_context: Optional[SolverInfo],
) -> None:
    """Copy cached dual values from a previous canonicalization onto new constraints.

    After a solve, dual values for auxiliary constraints are saved on
    ``expr._cached_aux_constraints``.  On re-canonicalization this function
    copies those saved duals onto the freshly created *new_constraints* so
    that the solver can use them as warm-start hints.

    The new constraints are always cached on the expression for the next
    round regardless of whether copying actually happens.
    """
    if solver_context is not None and solver_context.solver_supports_warm_start:
        old_constraints = getattr(expr, '_cached_aux_constraints', None)
        if old_constraints is not None and len(old_constraints) == len(new_constraints):
            for old_con, new_con in zip(old_constraints, new_constraints):
                for old_dv, new_dv in zip(old_con.dual_variables,
                                          new_con.dual_variables):
                    if old_dv.value is not None:
                        new_dv.save_value(old_dv.value)

    # Always cache the new constraints for the next solve.
    expr._cached_aux_constraints = new_constraints
