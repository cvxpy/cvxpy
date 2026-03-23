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

from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.elementwise.power import Power, PowerApprox
from cvxpy.atoms.elementwise.huber import huber

from cvxpy.reductions.dcp2cone.canonicalizers.quad.quad_form_canon import quad_form_canon
from cvxpy.reductions.dcp2cone.canonicalizers.quad.quad_over_lin_canon import quad_over_lin_canon
from cvxpy.reductions.dcp2cone.canonicalizers.quad.power_canon import power_canon
from cvxpy.reductions.dcp2cone.canonicalizers.quad.huber_canon import huber_canon

# Canonicalizations that return a quadratic objective (SymbolicQuadForm).
QUAD_CANON_METHODS = {
    quad_over_lin: quad_over_lin_canon,
    Power: power_canon,
    PowerApprox: power_canon,
    huber: huber_canon,
    QuadForm: quad_form_canon,
}
