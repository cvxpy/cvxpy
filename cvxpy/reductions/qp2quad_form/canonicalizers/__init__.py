"""
Copyright 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.atoms import *
from cvxpy.atoms.affine.index import special_index
from cvxpy.reductions.dcp2cone.canonicalizers import (
    CANON_METHODS as CONE_METHODS,)
from cvxpy.reductions.qp2quad_form.canonicalizers.huber_canon import *
from cvxpy.reductions.qp2quad_form.canonicalizers.power_canon import *
from cvxpy.reductions.qp2quad_form.canonicalizers.quad_form_canon import *
from cvxpy.reductions.qp2quad_form.canonicalizers.quad_over_lin_canon import *
from cvxpy.transforms.indicator import indicator

# TODO: remove pwl canonicalize methods, use EliminatePwl reduction instead

# Conic canonicalization methods.
CANON_METHODS = {}
CANON_METHODS[abs] = CONE_METHODS[abs]
CANON_METHODS[cumsum] = CONE_METHODS[cumsum]
CANON_METHODS[maximum] = CONE_METHODS[maximum]
CANON_METHODS[minimum] = CONE_METHODS[minimum]
CANON_METHODS[sum_largest] = CONE_METHODS[sum_largest]
CANON_METHODS[max] = CONE_METHODS[max]
CANON_METHODS[min] = CONE_METHODS[min]
CANON_METHODS[norm1] = CONE_METHODS[norm1]
CANON_METHODS[norm_inf] = CONE_METHODS[norm_inf]
CANON_METHODS[indicator] = CONE_METHODS[indicator]
CANON_METHODS[special_index] = CONE_METHODS[special_index]

# Canonicalizations that return a quadratic objective.
# Saved here for reference in other files.
QUAD_CANON_METHODS = {}
QUAD_CANON_METHODS[quad_over_lin] = quad_over_lin_canon
QUAD_CANON_METHODS[power] = power_canon
QUAD_CANON_METHODS[huber] = huber_canon
QUAD_CANON_METHODS[QuadForm] = quad_form_canon

CANON_METHODS.update(QUAD_CANON_METHODS)
