"""
Copyright 2025 CVXPY developers

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
from cvxpy.atoms import maximum
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.pnorm import Pnorm
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.reductions.expr2smooth.canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.expr2smooth.canonicalizers.pnorm_canon import pnorm_canon
from cvxpy.reductions.expr2smooth.canonicalizers.power_canon import power_canon
from cvxpy.reductions.expr2smooth.canonicalizers.maximum_canon import maximum_canon

CANON_METHODS = {
    abs: abs_canon,
    maximum : maximum_canon,
    # log: log_canon,
    power: power_canon,
    Pnorm : pnorm_canon,
    # inv: inv_pos_canon,
}
