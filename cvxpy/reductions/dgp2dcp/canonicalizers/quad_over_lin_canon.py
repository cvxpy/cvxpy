"""
Copyright, the CVXPY authors

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

from cvxpy.atoms.log_sum_exp import log_sum_exp


def quad_over_lin_canon(expr, args):
    if expr.args[1].is_scalar():
        # log(sum_axis(x^2) / y) = log_sum_exp(2*log(x), axis) - log(y)
        # Subtract scalar denominator once, outside the sum.
        numerator = log_sum_exp(2 * args[0], axis=expr.axis,
                                keepdims=expr.keepdims)
        return numerator - args[1], []
    # Non-scalar y: element-wise log(x_i^2/y_i) = 2*log(x_i) - log(y_i)
    ewise = 2 * args[0] - args[1]
    if expr.axis != ():
        return log_sum_exp(ewise, axis=expr.axis, keepdims=expr.keepdims), []
    return ewise, []
