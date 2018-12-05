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

from cvxpy.atoms import abs, maximum, max, minimum, norm1, norm_inf, sum_largest
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.maximum_canon import maximum_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.minimum_canon import minimum_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.max_canon import max_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.norm1_canon import norm1_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.norm_inf_canon import norm_inf_canon
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.sum_largest_canon import sum_largest_canon


CANON_METHODS = {
    abs: abs_canon,
    maximum: maximum_canon,
    max: max_canon,
    minimum: minimum_canon,
    norm1: norm1_canon,
    norm_inf: norm_inf_canon,
    sum_largest: sum_largest_canon
}
