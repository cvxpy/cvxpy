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
