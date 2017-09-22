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

from cvxpy.atoms import abs, hstack, norm2, reshape


def abs_canon(expr, reals, imags):
    # Imaginary.
    if reals[0] is None:
        output = abs(imags[0])
    elif imags[0] is None:  # Real
        output = abs(reals[0])
    else:  # Complex.
        real = reals[0].flatten()
        imag = imags[0].flatten()
        norms = norm2(hstack([real, imag]), axis=1)
        output = reshape(norms, expr.shape)
    return output, None
