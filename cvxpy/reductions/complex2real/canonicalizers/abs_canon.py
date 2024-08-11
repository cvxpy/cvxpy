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

from cvxpy.atoms import abs, pnorm, reshape, vstack


def abs_canon(expr, real_args, imag_args, real2imag):
    # Imaginary.
    if real_args[0] is None:
        output = abs(imag_args[0])
    elif imag_args[0] is None:  # Real
        output = abs(real_args[0])
    else:  # Complex.
        real = real_args[0].flatten(order='F')
        imag = imag_args[0].flatten(order='F')
        norms = pnorm(vstack([real, imag]), p=2, axis=0)
        output = reshape(norms, real_args[0].shape, order='F')
    return output, None
