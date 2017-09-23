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

from cvxpy.reductions.complex2real.atom_canonicalizers.abs_canon import abs_canon


def pnorm_canon(expr, real_args, imag_args, real2imag):
    abs_real_args, _ = abs_canon(expr, real_args, imag_args, real2imag)
    return expr.copy([abs_real_args]), None
