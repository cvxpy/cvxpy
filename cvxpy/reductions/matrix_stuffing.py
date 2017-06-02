"""
Copyright 2017 Robin Verschueren

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

import abc
import numpy as np

import cvxpy.settings as s
from cvxpy.reductions import Reduction, Solution


class MatrixStuffing(Reduction):
    __metaclass__ = abc.ABCMeta

    def accepts(self):
        return NotImplementedError

    def apply(self, problem):
        return NotImplementedError

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        var_map = inverse_data.var_offsets
        con_map = inverse_data.cons_id_map
        # Flip sign of opt val if maximize.
        opt_val = solution.opt_val
        if solution.status not in s.ERROR and not inverse_data.minimize:
            opt_val = -solution.opt_val

        if solution.status not in s.SOLUTION_PRESENT:
            return Solution(solution.status, opt_val, None, None)

        primal_vars, dual_vars = {}, {}

        # Split vectorized variable into components.
        x_opt = solution.primal_vars.values()[0]
        for var_id, offset in var_map.items():
            shape = inverse_data.var_shapes[var_id]
            size = np.prod(shape)
            primal_vars[var_id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
        # Remap dual variables.
        for old_con, new_con in con_map.items():
            dual_vars[old_con] = solution.dual_vars[new_con]

        return Solution(solution.status, opt_val, primal_vars, dual_vars)
