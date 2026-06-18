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

from cvxpy.constraints.complex_psd import ComplexPSD


def psd_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    if imag_args[0] is None:
        # Purely real: keep as regular PSD
        return [expr.copy([real_args[0]])], None
    else:
        if real_args[0] is None:
            real_args[0] = np.zeros(imag_args[0].shape)
        # Complex: create ComplexPSD, preserving constraint id
        return [ComplexPSD(real_args[0], imag_args[0], constr_id=expr.id)], None
