"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms import abs, vstack, pnorm, reshape


def abs_canon(expr, real_args, imag_args, real2imag):
    # Imaginary.
    if real_args[0] is None:
        output = abs(imag_args[0])
    elif imag_args[0] is None:  # Real
        output = abs(real_args[0])
    else:  # Complex.
        real = real_args[0].flatten()
        imag = imag_args[0].flatten()
        norms = pnorm(vstack([real, imag]), p=2, axis=0)
        output = reshape(norms, real_args[0].shape)
    return output, None
