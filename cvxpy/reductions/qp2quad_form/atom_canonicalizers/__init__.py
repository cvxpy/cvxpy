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
from cvxpy.transforms.indicator import indicator
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS as CONE_METHODS
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_over_lin_canon import *
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.power_canon import *
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_form_canon import *

CANON_METHODS = {}

# TODO: remove pwl canonicalize methods, use EliminatePwl reduction instead

# reuse cone canonicalization methods
CANON_METHODS[abs] = CONE_METHODS[abs]
CANON_METHODS[cumsum] = CONE_METHODS[cumsum]
CANON_METHODS[maximum] = CONE_METHODS[maximum]
CANON_METHODS[sum_largest] = CONE_METHODS[sum_largest]
CANON_METHODS[max] = CONE_METHODS[max]
CANON_METHODS[norm1] = CONE_METHODS[norm1]
CANON_METHODS[norm_inf] = CONE_METHODS[norm_inf]
CANON_METHODS[indicator] = CONE_METHODS[indicator]

# canonicalizations that are different for QPs
CANON_METHODS[quad_over_lin] = quad_over_lin_canon
CANON_METHODS[power] = power_canon
CANON_METHODS[QuadForm] = quad_form_canon
