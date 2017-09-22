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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import abs_canon


def norm1_canon(expr, args):
    x = args[0]
    axis = expr.axis

    # we need an absolute value constraint for the symmetric convex branches
    # (p >= 1)
    constraints = []
    # TODO(akshayka): Express this more naturally (recursively), in terms
    # of the other atoms
    abs_expr = abs(x)
    abs_x, abs_constraints = abs_canon(abs_expr, abs_expr.args)
    constraints += abs_constraints
    return sum(abs_x, axis=axis), constraints
