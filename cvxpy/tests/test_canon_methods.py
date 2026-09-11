"""
Copyright, the CVXPY authors

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

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.complex2real.canonicalizers import (
    CANON_METHODS as COMPLEX2REAL_METHODS,
)
from cvxpy.reductions.cone2cone.approx import ApproxCone2Cone
from cvxpy.reductions.cone2cone.exact import ExactCone2Cone
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS as DCP2CONE_METHODS
from cvxpy.reductions.dcp2cone.canonicalizers.quad import (
    QUAD_CANON_METHODS as QUAD_METHODS,
)
from cvxpy.reductions.dgp2dcp.canonicalizers import CANON_METHODS as DGP2DCP_METHODS
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import Valinvec2mixedint
from cvxpy.reductions.dnlp2smooth.canonicalizers import (
    SMOOTH_CANON_METHODS as DNLP2SMOOTH_METHODS,
)
from cvxpy.reductions.eliminate_pwl.canonicalizers import (
    CANON_METHODS as ELIMINATE_PWL_METHODS,
)

ALL_REGISTRIES = {
    "complex2real.CANON_METHODS": COMPLEX2REAL_METHODS,
    "cone2cone.ApproxCone2Cone.CANON_METHODS": ApproxCone2Cone.CANON_METHODS,
    "cone2cone.ExactCone2Cone.CANON_METHODS": ExactCone2Cone.CANON_METHODS,
    "dcp2cone.CANON_METHODS": DCP2CONE_METHODS,
    "dcp2cone.quad.QUAD_CANON_METHODS": QUAD_METHODS,
    "dgp2dcp.CANON_METHODS": DGP2DCP_METHODS,
    "discrete2mixedint.Valinvec2mixedint.CANON_METHODS": Valinvec2mixedint.CANON_METHODS,
    "dnlp2smooth.SMOOTH_CANON_METHODS": DNLP2SMOOTH_METHODS,
    "eliminate_pwl.CANON_METHODS": ELIMINATE_PWL_METHODS,
}


def test_canon_method_keys_are_classes() -> None:
    """Every canonicalizer registry key must be an Expression/Constraint class.

    Canonicalization looks up registries via type(expr), so a key that is a
    function rather than a class can never match: the atom silently falls
    through to default handling (e.g. the dgp2dcp registry once keyed the
    cp.quad_form *function* instead of the QuadForm class, producing wrong
    DGP results).
    """
    for name, registry in ALL_REGISTRIES.items():
        for key in registry:
            assert isinstance(key, type), f"{name} key {key!r} is not a class"
            assert issubclass(key, (Expression, Constraint)), (
                f"{name} key {key!r} is not an Expression or Constraint subclass"
            )
