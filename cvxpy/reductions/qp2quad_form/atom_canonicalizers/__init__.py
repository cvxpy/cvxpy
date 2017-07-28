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
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS as CONE_METHODS
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_over_lin_canon import *
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.power_canon import *
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_form_canon import *

CANON_METHODS = {}

# TODO: remove pwl canonicalize methods, use EliminatePwl reduction instead

# reuse cone canonicalization methods
CANON_METHODS[affine_prod] = CONE_METHODS[affine_prod]
CANON_METHODS[abs] = CONE_METHODS[abs]
CANON_METHODS[cumsum] = CONE_METHODS[cumsum]
CANON_METHODS[max_elemwise] = CONE_METHODS[max_elemwise]
CANON_METHODS[sum_largest] = CONE_METHODS[sum_largest]
CANON_METHODS[max_entries] = CONE_METHODS[max_entries]
CANON_METHODS[pnorm] = CONE_METHODS[pnorm]

# canonicalizations that are different for QPs
CANON_METHODS[quad_over_lin] = quad_over_lin_canon
CANON_METHODS[power] = power_canon
CANON_METHODS[QuadForm] = quad_form_canon
