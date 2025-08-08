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

from cvxpy.constraints.cones import Cone
from cvxpy.utilities import scopes


# TODO: write docstrings for this
class LogDet(Cone):
    """
    A constraint of the form :math:`\\log \\det(X) \\geq 0`
    """

    def __init__(self, expr, constr_id=None) -> None:
        # Argument must be square matrix.
        if len(expr.shape) != 2 or expr.shape[0] != expr.shape[1]:
            raise ValueError(
                "Non-square matrix in positive definite constraint."
            )
        super(LogDet, self).__init__([expr], constr_id)

    def name(self) -> str:
        return "LogDet(%s)" % self.args[0]

    def is_dcp(self, dpp: bool = False) -> bool:
        """A LogDet constraint is DCP if the constrained expression is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    # TODO implement these
    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        NumPy.ndarray
        """
        raise NotImplementedError(
            "LogDet residual is not implemented yet."
        )

    def _dual_cone(self, *args):
        """Implements the dual cone of the LogDet cone"""
        raise NotImplementedError(
            "Dual cone of LogDet is not implemented yet."
        )
