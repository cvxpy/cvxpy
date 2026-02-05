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

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from cvxpy.atoms.atom import Atom
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
    - Computing the value raises an exception

    Otherwise, returns a dense ndarray matching expr.shape.
    """
    if solver_context is None or not solver_context.solver_supports_warm_start:
        return None
    try:
        val = expr.value
    except Exception:
        return None
    if val is None:
        return None
    return np.atleast_1d(np.array(val, dtype=float)).reshape(expr.shape)
