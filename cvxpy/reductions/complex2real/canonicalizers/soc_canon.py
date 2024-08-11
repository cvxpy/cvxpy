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

from cvxpy.atoms import reshape, vstack
from cvxpy.constraints import SOC
from cvxpy.expressions.variable import Variable


def soc_canon(expr, real_args, imag_args, real2imag):
    # Imaginary.
    if real_args[1] is None:
        output = [SOC(real_args[0], imag_args[1],
                      axis=expr.axis, constr_id=real2imag[expr.id])]
    elif imag_args[1] is None:  # Real
        output = [SOC(real_args[0], real_args[1],
                      axis=expr.axis, constr_id=expr.id)]
    else:  # Complex.
        orig_shape = real_args[1].shape
        real = real_args[1].flatten(order='F')
        imag = imag_args[1].flatten(order='F')
        flat_X = Variable(real.shape)
        inner_SOC = SOC(flat_X,
                        vstack([real, imag]),
                        axis=0)
        real_X = reshape(flat_X, orig_shape, order='F')
        outer_SOC = SOC(real_args[0], real_X,
                        axis=expr.axis, constr_id=expr.id)
        output = [inner_SOC, outer_SOC]
    return output, None
