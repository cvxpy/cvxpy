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

from cvxpy.expressions.variable import Variable


def variable_canon(expr, real_args, imag_args, real2imag):
    if expr.is_real():
        return expr, None

    imag = Variable(expr.shape, var_id=real2imag[expr.id])
    if expr.is_imag():
        return None, imag
    elif expr.is_complex() and expr.is_hermitian():
        return Variable(expr.shape, var_id=expr.id, symmetric=True), (imag - imag.T)/2
    else:  # Complex.
        return Variable(expr.shape, var_id=expr.id), imag
