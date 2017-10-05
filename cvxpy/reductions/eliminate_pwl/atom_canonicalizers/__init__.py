"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms import abs, maximum, max, norm1, norm_inf, sum_largest
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.maximum_canon import maximum_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.max_canon import max_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.norm1_canon import norm1_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.norm_inf_canon import norm_inf_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.sum_largest_canon import sum_largest_canon


CANON_METHODS = {
    abs: abs_canon,
    maximum: maximum_canon,
    max: max_canon,
    norm1: norm1_canon,
    norm_inf: norm_inf_canon,
    sum_largest: sum_largest_canon
}
