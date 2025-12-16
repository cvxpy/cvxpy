"""
Copyright 2013 Steven Diamond

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

import cvxpy.settings as s

INF_OR_UNB_MESSAGE = """
    The problem is either infeasible or unbounded, but the solver
    cannot tell which. Disable any solver-specific presolve methods
    and re-solve to determine the precise problem status.

    For GUROBI and CPLEX you can automatically perform this re-solve
    with the keyword argument prob.solve(reoptimize=True, ...).
    """


def failure_solution(status, attr=None) -> "Solution":
    """Factory function for infeasible or unbounded solutions.

    Parameters
    ----------
    status : str
        The problem status.

    Returns
    -------
    Solution
        A solution object.
    """
    if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
        opt_val = np.inf
    elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
        opt_val = -np.inf
    else:
        opt_val = None
    if attr is None:
        attr = {}
    if status == s.INFEASIBLE_OR_UNBOUNDED:
        attr['message'] = INF_OR_UNB_MESSAGE
    return Solution(status, opt_val, {}, {}, attr)


class Solution:
    """A solution to an optimization problem.

    Attributes
    ----------
    status : str
        The status code.
    opt_val : float
        The optimal value.
    primal_vars : dict of id to NumPy ndarray
        A map from variable ids to optimal values.
    dual_vars : dict of id to NumPy ndarray
        A map from constraint ids to dual values.
    attr : dict
        Miscelleneous information propagated up from a solver.
    """
    def __init__(self, status, opt_val, primal_vars, dual_vars, attr) -> None:
        self.status = status
        self.opt_val = opt_val
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.attr = attr

    def copy(self) -> "Solution":
        return Solution(self.status,
                        self.opt_val,
                        self.primal_vars,
                        self.dual_vars,
                        self.attr)

    def __str__(self) -> str:
        return "Solution(status=%s, opt_val=%s, primal_vars=%s, dual_vars=%s, attr=%s)" % (
          self.status, self.opt_val, self.primal_vars, self.dual_vars, self.attr)

    def __repr__(self) -> str:
        return "Solution(%s, %s, %s, %s)" % (self.status,
                                             self.primal_vars,
                                             self.dual_vars,
                                             self.attr)

    @property
    def batch_shape(self) -> tuple:
        """Return the batch shape from attr, or () if not batched."""
        return self.attr.get(s.BATCH_SHAPE, ())

    @property
    def is_batched(self) -> bool:
        """Return True if this is a batched solution."""
        return len(self.batch_shape) > 0

    def has_solution(self) -> bool:
        """Check if any solution is present (status in SOLUTION_PRESENT)."""
        if self.is_batched:
            return bool(np.any(np.isin(self.status, list(s.SOLUTION_PRESENT))))
        return self.status in s.SOLUTION_PRESENT

    def has_inf_or_unb(self) -> bool:
        """Check if any solution is infeasible or unbounded."""
        if self.is_batched:
            return bool(np.any(np.isin(self.status, list(s.INF_OR_UNB))))
        return self.status in s.INF_OR_UNB

    def has_error(self) -> bool:
        """Check if any solution has an error."""
        if self.is_batched:
            return bool(np.any(np.isin(self.status, list(s.ERROR))))
        return self.status in s.ERROR

    def all_not_error(self) -> bool:
        """Check if all solutions are not errors (for opt_val negation)."""
        if self.is_batched:
            return bool(np.all(~np.isin(self.status, list(s.ERROR))))
        return self.status not in s.ERROR

    def has_inaccurate(self) -> bool:
        """Check if any solution is inaccurate."""
        if self.is_batched:
            return bool(np.any(np.isin(self.status, list(s.INACCURATE))))
        return self.status in s.INACCURATE

    def has_infeasible_or_unbounded(self) -> bool:
        """Check if any solution has INFEASIBLE_OR_UNBOUNDED status."""
        if self.is_batched:
            return bool(np.any(self.status == s.INFEASIBLE_OR_UNBOUNDED))
        return self.status == s.INFEASIBLE_OR_UNBOUNDED
