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


import numpy as np

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
        # Argument must be square matrix (possibly batched).
        if len(expr.shape) < 2 or expr.shape[-2] != expr.shape[-1]:
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

    def num_cones(self) -> int:
        """The number of PSD cones."""
        return int(np.prod(self.args[0].shape[:-2])) if len(self.args[0].shape) > 2 else 1

    def _cone_size(self) -> int:
        """The dimension of each PSD cone (the matrix side length n)."""
        return int(self.args[0].shape[-1])

    def cone_sizes(self) -> list[int]:
        """The dimensions of the PSD cones."""
        return [self._cone_size()] * self.num_cones()

    @property
    def size(self) -> int:
        """The number of entries in the combined cones."""
        return self._cone_size() ** 2 * self.num_cones()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        NumPy.ndarray
        """
        if self.expr.value is None:
            return None
        # Deferred import to avoid circular import:
        # psd -> transpose -> affine_atom -> atom -> expression -> psd
        from cvxpy.atoms.affine.transpose import swapaxes as cp_swapaxes
        min_eig = cvxtypes.lambda_min()(
            self.args[0] + cp_swapaxes(self.args[0], -2, -1))/2
        return cvxtypes.neg()(min_eig).value

    def _dual_cone(self, *args):
        """Implements the dual cone of the PSD cone See Pg 85 of the
        MOSEK modelling cookbook for more information"""
        if not args:
            return self.dual_variables[0] >> 0
        else:
            # some assertions for verifying `args`
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return args[0] >> 0


class SvecPSD(Cone):
    """A PSD constraint in scaled vectorized (svec) form.

    The argument is a 1-D expression of length ``n * (n + 1) // 2``
    representing the scaled lower- or upper-triangular entries of a
    symmetric positive semidefinite matrix.  This constraint is produced
    automatically by the :class:`ExactCone2Cone` reduction when a solver
    requires the triangular PSD representation.

    Parameters
    ----------
    expr : Expression
        A 1-D affine expression of length ``n * (n + 1) // 2``.
    n : int
        The side length of the original PSD matrix.
    constr_id : int, optional
        A unique id for the constraint.
    """

    def __init__(self, expr, n: int, constr_id=None) -> None:
        self._n = n
        super().__init__([expr], constr_id)

    def get_data(self):
        """Data needed to copy."""
        return [self._n, self.id]

    def name(self) -> str:
        return "svec_psd(%s, n=%d)" % (self.args[0], self._n)

    def is_dcp(self, dpp: bool = False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def num_cones(self) -> int:
        tri_dim = self._n * (self._n + 1) // 2
        return self.args[0].size // tri_dim

    def _cone_size(self) -> int:
        """The matrix side length (not the triangular dimension)."""
        return self._n

    def cone_sizes(self) -> list[int]:
        return [self._n] * self.num_cones()

    @property
    def size(self) -> int:
        return self._n * (self._n + 1) // 2 * self.num_cones()

    @property
    def residual(self):
        if self.expr.value is None:
            return None
        raise NotImplementedError(
            "Residual is not implemented for SvecPSD. "
            "Check the residual on the original PSD constraint instead."
        )

    def _dual_cone(self, *args):
        """The dual of the PSD cone is itself."""
        if not args:
            return SvecPSD(self.dual_variables[0], n=self._n)
        else:
            assert len(args) == len(self.args)
            return SvecPSD(args[0], n=self._n)
