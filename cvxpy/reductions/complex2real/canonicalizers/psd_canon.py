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

import numpy as np

from cvxpy.atoms import bmat


def complex_psd_matrix(real_part, imag_part):
    """Return the real block representation of a complex PSD expression."""
    if imag_part is None:
        return real_part
    if real_part is None:
        real_part = np.zeros(imag_part.shape)
    return bmat([[real_part, -imag_part],
                 [imag_part, real_part]])


def psd_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    matrix = complex_psd_matrix(real_args[0], imag_args[0])
    return [expr.copy([matrix])], None
