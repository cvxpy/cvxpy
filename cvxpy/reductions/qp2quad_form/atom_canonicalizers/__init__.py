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
