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


from cvxpy.atoms import *
from cvxpy.expressions.variable import Variable
from abs_canon import *
from maximum_canon import *
from max_canon import *
from norm1_canon import *
from norm_inf_canon import *
from sum_largest_canon import *


CANON_METHODS = {
    abs : abs_canon,
    maximum : maximum_canon,
    max : max_canon,
    norm1 : norm1_canon,
    norm_inf : norm1_canon,
    sum_largest : sum_largest_canon
}
