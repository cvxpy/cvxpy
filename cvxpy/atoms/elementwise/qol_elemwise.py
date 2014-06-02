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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.constraints import SOC_Elemwise
import scipy.sparse as sp

def qol_elemwise(arg_objs, size, data=None):
    """Reduces the atom to an affine expression and list of constraints.

    Parameters
    ----------
    arg_objs : list
        LinExpr for each argument.
    size : tuple
        The size of the resulting expression.
    data :
        Additional data required by the atom.

    Returns
    -------
    tuple
        (LinOp for objective, list of constraints)
    """
    x = arg_objs[0]
    y = arg_objs[1]
    t = lu.create_var(x.size)
    two = lu.create_const(2, (1, 1))
    constraints = [SOC_Elemwise(lu.sum_expr([y, t]),
                                [lu.sub_expr(y, t),
                                 lu.mul_expr(two, x, x.size)]),
                   lu.create_geq(y)]
    return (t, constraints)
