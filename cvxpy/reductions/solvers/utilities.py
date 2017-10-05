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

import cvxpy.interface as intf


def extract_dual_value(result_vec, offset, constraint):
    value = result_vec[offset:offset + constraint.size]
    if constraint.size == 1:
        value = intf.scalar_value(value)
    offset += constraint.size
    return value, offset


def get_dual_values(result_vec, parse_func, constraints):
    """Gets the values of the dual variables.

    Parameters
    ----------
    result_vec : array_like
        A vector containing the dual variable values.
    parse_func : function
        A function that extracts a dual value from the result vector
        for a particular constraint. The function should accept
        three arguments: the result vector, an offset, and a
        constraint, in that order. An example of a parse_func is
        extract_dual_values, defined in this module. Some solvers
        may need to implement their own parse functions.
    constraints : list
        A list of the constraints in the problem.

    Returns
    -------
       A map of constraint id to dual variable value.
    """
    dual_vars = {}
    offset = 0
    for constr in constraints:
        # TODO reshape based on dual variable size.
        dual_vars[constr.id], offset = parse_func(result_vec, offset, constr)
    return dual_vars
