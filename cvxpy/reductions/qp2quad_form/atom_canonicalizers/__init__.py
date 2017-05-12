from cvxpy.atoms import *
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS as CONE_METHODS
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_over_lin_canon import *
from cvxpy.reductions.qp2quad_form.atom_canonicalizers.power_canon import *

CANON_METHODS = {}

# reuse cone canonicalization methods
CANON_METHODS[affine_prod] = CONE_METHODS[affine_prod]
CANON_METHODS[abs] = CONE_METHODS[abs]
CANON_METHODS[max_elemwise] = CONE_METHODS[max_elemwise]
CANON_METHODS[sum_largest] = CONE_METHODS[sum_largest]
CANON_METHODS[max_entries] = CONE_METHODS[max_entries]
CANON_METHODS[pnorm] = CONE_METHODS[pnorm]

# canonicalizations that are different for QPs
CANON_METHODS[quad_over_lin] = quad_over_lin_canon
CANON_METHODS[power] = power_canon