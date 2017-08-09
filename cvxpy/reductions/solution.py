"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

import cvxpy.settings as s


def failure_solution(status):
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
    if status == s.INFEASIBLE:
        opt_val = np.inf
    elif status == s.UNBOUNDED:
        opt_val = -np.inf
    else:
        opt_val = None
    return Solution(status, opt_val, {}, {}, {})


class Solution(object):
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
    def __init__(self, status, opt_val, primal_vars, dual_vars, attr):
        self.status = status
        self.opt_val = opt_val
        self.primal_vars = primal_vars
        self.dual_vars = dual_vars
        self.attr = attr

    def copy(self):
        return Solution(self.status,
                        self.opt_val,
                        self.primal_vars,
                        self.dual_vars,
                        self.attr)

    def __str__(self):
        return "Solution(%s, %s, %s, %s)" % (self.status,
                                             self.primal_vars,
                                             self.dual_vars,
                                             self.attr)
