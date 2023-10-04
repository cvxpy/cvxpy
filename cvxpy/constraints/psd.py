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

from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes


class PSD(Cone):
    """A constraint of the form :math:`\\frac{1}{2}(X + X^T) \\succcurlyeq_{S_n^+} 0`

    Applying a ``PSD`` constraint to a two-dimensional expression ``X``
    constrains its symmetric part to be positive semidefinite: i.e.,
    it constrains ``X`` to be such that

    .. math::

        z^T(X + X^T)z \\geq 0,

    for all :math:`z`.

    The preferred way of creating a ``PSD`` constraint is through operator
    overloading. To constrain an expression ``X`` to be PSD, write
    ``X >> 0``; to constrain it to be negative semidefinite, write
    ``X << 0``. Strict definiteness constraints are not provided,
    as they do not make sense in a numerical setting.

    Parameters
    ----------
    expr : Expression.
        The expression to constrain; *must* be two-dimensional.
    constr_id : int
        A unique id for the constraint.
    """

    def __init__(self, expr, constr_id=None) -> None:
        # Argument must be square matrix.
        if len(expr.shape) != 2 or expr.shape[0] != expr.shape[1]:
            raise ValueError(
                "Non-square matrix in positive definite constraint."
            )
        super(PSD, self).__init__([expr], constr_id)

    def name(self) -> str:
        return "%s >> 0" % self.args[0]

    def is_dcp(self, dpp: bool = False) -> bool:
        """A PSD constraint is DCP if the constrained expression is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        NumPy.ndarray
        """
        if self.expr.value is None:
            return None
        min_eig = cvxtypes.lambda_min()(self.args[0] + self.args[0].T)/2
        return cvxtypes.neg()(min_eig).value

    def _dual_cone(self, *args):
        """Implements the dual cone of the PSD cone See Pg 85 of the
        MOSEK modelling cookbook for more information"""
        if args is None:
            return self.dual_variables[0] >> 0
        else:
            # some assertions for verifying `args`
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return args[0] >> 0
