"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc

from cvxpy.constraints.constraint import Constraint


class Cone(Constraint):
    """A base class for all conic constraints in CVXPY

    These are special constraints imposing set membership in convex cones
    CVXPY supports modelling using the following cones as of today:
    - ExpCone
    - SOC (Second Order Cone)
    - PowCone3D
    - PowConeND
    - RelEntrConeQuad
    - OpRelEntrConeQuad
    - PSD/NSD

    Parameters
    ----------
    args : list
        A list of expression trees.
    constr_id : int
        A unique id for the constraint.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, args, constr_id=None) -> None:
        super(Cone, self).__init__(args, constr_id)

    def _dual_cone(self, args):
        """Method for modelling problems with the dual cone of `Cone`

        If the user simply calls the method without any arguments, then
        the dual cone to the current instance of `Cone` will be returned, else,
        the user can also choose to freely model with the dual cone of `Cone`
        by constraining aribitrary expressions in the same.

        Will typically not be used stand-alone, and would be
        indirectly accessed via `dual_residual` which will return
        the violation of CVXPY's computed dual variables w.r.t. `Cone`'s
        dual-cone"""
        raise NotImplementedError

    @property
    def dual_residual(self) -> float:
        """Computes the residual (see Constraint.violation for a
        more formal definition) for the dual cone of the current instance
        of `Cone` w.r.t. the recovered dual variables

        Primarily intended to be used for KKT checks
        """
        return self._dual_cone(*self.dual_variables).residual
