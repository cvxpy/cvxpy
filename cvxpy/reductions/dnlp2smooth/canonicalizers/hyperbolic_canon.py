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
from cvxpy.utilities.bounds import get_expr_bounds


def atanh_canon(expr, args):
    domain_bounds = [-1, 1]
    expr_bounds = get_expr_bounds(args[0])
    if expr_bounds is not None:
        domain_bounds = [np.maximum(domain_bounds[0], expr_bounds[0]),
                         np.minimum(domain_bounds[1], expr_bounds[1])]
    t = Variable(args[0].shape, bounds=domain_bounds)
    if args[0].value is not None:
        t.value = args[0].value
    return expr.copy([t]), [t == args[0]]
