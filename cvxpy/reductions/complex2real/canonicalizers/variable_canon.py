"""
Copyright 2013 Steven Diamond, 2022 - the CVXPY Authors.

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

from cvxpy.atoms.affine.upper_tri import vec_to_upper_tri
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable


def variable_canon(expr, real_args, imag_args, real2imag):
    if expr.is_real():
        # Purely real
        return expr, None
    elif expr.is_imag():
        # Purely imaginary
        imag = Variable(expr.shape, var_id=real2imag[expr.id])
        return None, imag
    elif expr.is_complex() and expr.is_hermitian():
        n = expr.shape[0]
        real = Variable((n, n), var_id=expr.id, symmetric=True)
        if n > 1:
            imag_var = Variable(shape=n*(n-1)//2, var_id=real2imag[expr.id])
            imag_upper_tri = vec_to_upper_tri(imag_var, strict=True)
            imag = skew_symmetric_wrap(imag_upper_tri - imag_upper_tri.T)
        else:
            imag = Constant([[0.0]])
        return real, imag
    else:
        # General complex.
        real = Variable(expr.shape, var_id=expr.id)
        imag = Variable(expr.shape, var_id=real2imag[expr.id])
        return real, imag
