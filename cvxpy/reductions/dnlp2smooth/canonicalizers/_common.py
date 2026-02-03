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

import numpy as np

from cvxpy.expressions.variable import Variable


def canonicalize_unary_smooth(expr, args, bounds=None, default_value=None):
    """
    Generic canonicalization for smooth unary functions.

    Ensures the argument is a Variable. If already a Variable (with no bounds
    constraint), returns unchanged. Otherwise, creates a new Variable with
    an equality constraint to the original argument.

    Parameters
    ----------
    expr : Atom
        The expression being canonicalized.
    args : list
        The canonicalized arguments of expr (expects single argument).
    bounds : tuple or None
        Optional (lower, upper) bounds for the new Variable.
    default_value : array-like or None
        Default value to use if arg has no value. If None, uses arg.value.

    Returns
    -------
    tuple
        (canonicalized_expr, list_of_constraints)
    """
    arg = args[0]

    # If already a Variable with no bounds requirement, return unchanged
    if isinstance(arg, Variable) and bounds is None:
        return expr.copy([arg]), []

    # Create new Variable, optionally with bounds
    if bounds is not None:
        t = Variable(arg.shape, bounds=bounds)
    else:
        t = Variable(arg.shape)

    # Set initial value
    if arg.value is not None:
        t.value = arg.value
    elif default_value is not None:
        if callable(default_value):
            t.value = default_value(arg.shape)
        else:
            t.value = np.broadcast_to(default_value, arg.shape)

    return expr.copy([t]), [t == arg]
