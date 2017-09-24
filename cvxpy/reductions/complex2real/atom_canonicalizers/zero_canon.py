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

def zero_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is None:
        return expr.copy(real_args), None
    elif real_args[0] is None:
        return None, expr.copy(imag_args)
    else:
        return expr.copy(real_args), expr.copy(imag_args)
