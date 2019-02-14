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

from cvxpy.expressions.constants import CallbackParam
import numpy as np


def param_canon(expr, real_args, imag_args, real2imag):
    if expr.is_real():
        return expr, None
    elif expr.is_imag():
        imag = CallbackParam(lambda: np.imag(expr.value), expr.shape)
        return None, imag
    else:
        real = CallbackParam(lambda: np.real(expr.value), expr.shape)
        imag = CallbackParam(lambda: np.imag(expr.value), expr.shape)
        return (real, imag)
