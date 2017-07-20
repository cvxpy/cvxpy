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
import cvxpy.settings as s


class Solution(object):
    """A solution object.

    Attributes:
        status: status code
        opt_val: float
        primal_vars: dict of id to NumPy ndarray
        dual_vars: dict of id to NumPy ndarray
        attr: dict of other attributes.
    """
    def __init__(self, status=None, opt_val=None, primal_vars=None,
                 dual_vars=None, attr=None):
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
