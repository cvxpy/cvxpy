"""
Copyright 2013 Steven Diamond

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
from typing import List

from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes

"""
Rotated Second Order Cone constraint.

Implements:

    2 y z >= ||x||_2^2
    y >= 0
    z >= 0
"""




class RSOC(Cone):
    """
    Rotated Second Order Cone:

        2 y z >= ||x||^2
        y >= 0, z >= 0
    """

    def __init__(self, x, y, z, constr_id=None) -> None:
        x = cvxtypes.expression().cast_to_const(x)
        y = cvxtypes.expression().cast_to_const(y)
        z = cvxtypes.expression().cast_to_const(z)

        if not y.is_real() or not z.is_real():
            raise ValueError("y and z must be real.")

        super(RSOC, self).__init__([x, y, z], constr_id)

    def __str__(self) -> str:
        return "RSOC(%s, %s, %s)" % (self.args[0], self.args[1], self.args[2])

    def is_dcp(self, dpp: bool = False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def canonicalize(self):
        """
        Convert RSOC into SOC:

            || [2x ; y - z] ||_2 <= y + z
        """

        # 🔴 Import locally to avoid circular import
        from cvxpy.atoms.affine.vstack import vstack
        from cvxpy.constraints.second_order import SOC

        x, y, z = self.args

        soc_vec = vstack([2 * x, y - z])
        soc_rhs = y + z

        soc_con = SOC(soc_rhs, soc_vec)
        canon, aux = soc_con.canonical_form
        nonneg_y = (y >= 0).canonical_form
        nonneg_z = (z >= 0).canonical_form
        return canon, aux + [nonneg_y[0]] + nonneg_y[1] + [nonneg_z[0]] + nonneg_z[1]

    def num_cones(self):
        return 1

    def cone_sizes(self) -> List[int]:
        x = self.args[0]
        return [1 + x.size]

    @property
    def size(self) -> int:
        return self.cone_sizes()[0]

    @property
    def residual(self):
        """
        Compute residual by converting to equivalent SOC
        and using its residual.
        """
        from cvxpy.atoms.affine.vstack import vstack
        from cvxpy.constraints.second_order import SOC

        x, y, z = self.args

        if any(arg.value is None for arg in self.args):
            return None

        soc_vec = vstack([2 * x, y - z])
        soc_rhs = y + z

        soc_con = SOC(soc_rhs, soc_vec)

        return soc_con.residual

