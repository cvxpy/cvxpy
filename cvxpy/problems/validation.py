"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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

from collections.abc import Iterable

import numpy as np

from cvxpy.constraints.constraint import Constraint


def _constraint_violation_to_scalar(constraint: Constraint) -> float:
    """Return a scalar violation value for a constraint."""
    violation = constraint.violation()
    violation_arr = np.asarray(violation)
    if violation_arr.size == 0:
        return 0.0

    return float(np.linalg.norm(violation_arr.ravel(), ord=2))


def _max_constraint_violation(constraints: Iterable[Constraint]) -> float:
    """Return the maximum scalar violation over a collection of constraints."""
    max_violation = 0.0

    for constraint in constraints:
        max_violation = max(
            max_violation,
            _constraint_violation_to_scalar(constraint),
        )

    return max_violation
