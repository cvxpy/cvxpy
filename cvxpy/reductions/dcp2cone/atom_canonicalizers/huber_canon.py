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

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.dcp2cone.atom_canonicalizers.power_canon import power_canon


def huber_canon(expr, args):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(shape)
    s = Variable(shape)

    # n**2 + 2*M*|s|
    # TODO(akshayka): Make use of recursion inherent to canonicalization
    # process and just return a power / abs expressions for readability sake
    power_expr = power(n, 2)
    n2, constr_sq = power_canon(power_expr, power_expr.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)
    obj = n2 + 2 * M * abs_s

    # x == s + n
    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)
    return obj, constraints
