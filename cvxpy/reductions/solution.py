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
