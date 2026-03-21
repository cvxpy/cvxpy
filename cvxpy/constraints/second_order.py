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

from typing import List, Optional

import numpy as np

from cvxpy.constraints.cones import Cone
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes


class SOC(Cone):
    """A second-order cone constraint for each row/column.

    Assumes ``t`` is a vector the same length as ``X``'s columns (rows) for
    ``axis == 0`` (``1``).

    Attributes:
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.
    """

    def __init__(self, t, X, axis: int = 0, constr_id=None) -> None:
        t = cvxtypes.expression().cast_to_const(t)
        if len(t.shape) >= 2 or not t.is_real():
            raise ValueError("Invalid first argument.")
        # Check t has one entry per cone.
        if (len(X.shape) <= 1 and t.size > 1) or \
           (len(X.shape) == 2 and t.size != X.shape[1-axis]) or \
           (len(X.shape) == 1 and axis == 1):
            raise ValueError(
                "Argument dimensions %s and %s, with axis=%i, are incompatible."
                % (t.shape, X.shape, axis)
            )
        self.axis = axis
        if len(t.shape) == 0:
            t = t.flatten(order='F')
        super(SOC, self).__init__([t, X], constr_id)

    def __str__(self) -> str:
        return "SOC(%s, %s)" % (self.args[0], self.args[1])

    @property
    def residual(self) -> Optional[np.ndarray]:
        """
        For each cone, returns:

        ||(t,X) - proj(t,X)||
        with
        proj(t,X) = (t,X)                       if t >= ||x||
                    0.5*(t/||x|| + 1)(||x||,x)  if -||x|| < t < ||x||
                    0                           if t <= -||x||

        References:
             https://docs.mosek.com/modeling-cookbook/practical.html#distance-to-a-cone
             https://math.stackexchange.com/questions/2509986/projection-onto-the-second-order-cone
        """

        t = self.args[0].value
        X = self.args[1].value
        if t is None or X is None:
            return None

        # Convert scalars to arrays for uniform handling
        t = np.atleast_1d(t)
        X = np.atleast_1d(X)

        # Reduce axis = 0 to axis = 1.
        if self.axis == 0:
            X = X.T

        promoted = X.ndim == 1
        X = np.atleast_2d(X)

        # Initializing with zeros makes "0 if t <= -||x||" the default case for the projection
        t_proj = np.zeros(t.shape)
        X_proj = np.zeros(X.shape)

        norms = np.linalg.norm(X, ord=2, axis=1)

        # 1. proj(t,X) = (t,X) if t >= ||x||
        t_geq_x_norm = t >= norms
        t_proj[t_geq_x_norm] = t[t_geq_x_norm]
        X_proj[t_geq_x_norm] = X[t_geq_x_norm]

        # 2. proj(t,X) = 0.5*(t/||x|| + 1)(||x||,x)  if -||x|| < t < ||x||
        abs_t_less_x_norm = np.abs(t) < norms
        avg_coeff = 0.5 * (1 + t/norms)
        X_proj[abs_t_less_x_norm] = avg_coeff[abs_t_less_x_norm, None] * X[abs_t_less_x_norm]
        t_proj[abs_t_less_x_norm] = avg_coeff[abs_t_less_x_norm] * norms[abs_t_less_x_norm]

        Xt = np.concatenate([X, t[:, None]], axis=1)
        Xt_proj = np.concatenate([X_proj, t_proj[:, None]], axis=1)
        resid = np.linalg.norm(Xt - Xt_proj, ord=2, axis=1)

        # Demote back to 1D.
        if promoted:
            return resid[0]
        else:
            return resid

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.axis, self.id]

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.args[0].size

    def _cone_size(self) -> int:
        """The size of each second-order cone (1 + dimension of X).
        """
        X = self.args[1]
        if len(X.shape) == 0:
            X_dim = 1
        else:
            # X is 2D or 1D with axis = 0.
            X_dim = X.shape[self.axis]
        return 1 + X_dim

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        return self._cone_size() * self.num_cones()

    def cone_sizes(self) -> List[int]:
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        return [self._cone_size()] * self.num_cones()

    def is_dcp(self, dpp: bool = False) -> bool:
        """An SOC constraint is DCP if each of its arguments is affine.
        """
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def save_dual_value(self, value) -> None:
        cone_size = self._cone_size()
        value = np.reshape(value, (-1, cone_size))
        t = value[:, 0]
        X = value[:, 1:]
        if len(self.args[1].shape) == 0:
            # Scalar X: extract scalar from 2D array
            X = X[0, 0]
        elif self.axis == 0:
            X = X.T
        self.dual_variables[0].save_value(t)
        self.dual_variables[1].save_value(X)

    def _dual_cone(self, *args):
        """Implements the dual cone of the second-order cone
        See Pg 85 of the MOSEK modelling cookbook for more information"""
        if not args:
            return SOC(self.dual_variables[0], self.dual_variables[1], self.axis)
        else:
            # some assertions for verifying `args`
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return SOC(args[0], args[1], self.axis)


class RSOC(Cone):
    """A rotated second-order cone constraint.

    Represents the constraint:

        2*y*z >= ||x||_2^2,  y >= 0,  z >= 0

    where x is a vector and y, z are scalars. Supports batching:
    if X is a matrix, y and z are vectors, the constraint is applied
    column-wise to each column of X.

    Parameters
    ----------
    X : Expression
        The vector (or matrix) part of the constraint.
    y : Expression
        The first scalar part (or vector for batched constraints).
    z : Expression
        The second scalar part (or vector for batched constraints).
    axis : int
        Axis along which to apply the constraint (0 = column-wise, 1 = row-wise).
    """

    def __init__(self, X, y, z, axis: int = 0, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        X = Expression.cast_to_const(X)
        y = Expression.cast_to_const(y)
        z = Expression.cast_to_const(z)

        if not y.is_real() or not z.is_real():
            raise ValueError("y and z must be real.")

        # Scalar case
        if y.ndim == 0 or y.size == 1:
            if z.size != 1:
                raise ValueError("y and z must have the same shape.")
        else:
            # Batched case: y and z must have the same shape
            if y.shape != z.shape:
                raise ValueError("y and z must have the same shape.")
            # X must be 2D with matching number of columns/rows
            if X.ndim < 2:
                raise ValueError(
                    "X must be a matrix for batched RSOC constraints.")
            n_cones = y.size
            if axis == 0 and X.shape[1] != n_cones:
                raise ValueError(
                    "Number of columns of X must match size of y and z.")
            if axis == 1 and X.shape[0] != n_cones:
                raise ValueError(
                    "Number of rows of X must match size of y and z.")

        self.axis = axis
        super(RSOC, self).__init__([X, y, z], constr_id)

    def __str__(self) -> str:
        return "RSOC(%s, %s, %s)" % (self.args[0], self.args[1], self.args[2])

    def num_cones(self):
        """The number of RSOC constraints (1 for scalar, n for batched)."""
        return self.args[1].size

    @property
    def residual(self):
        """Returns the residual of the constraint."""
        X = self.args[0].value
        y = self.args[1].value
        z = self.args[2].value
        if X is None or y is None or z is None:
            return None
        X = np.atleast_1d(np.array(X, dtype=float))
        y = np.atleast_1d(np.array(y, dtype=float)).ravel()
        z = np.atleast_1d(np.array(z, dtype=float)).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.axis == 1:
            X = X.T
        # After any transpose, X is (n_x, n_cones); sum over axis=0 gives (n_cones,)
        # But if axis==1 transposed to (n_cones, n_x), sum over axis=1 gives (n_cones,)
        sum_axis = 1 if (self.axis == 1) else 0
        norms_sq = np.sum(X ** 2, axis=sum_axis)  # shape (n_cones,)
        lhs = 2 * y * z
        viol_cone = norms_sq - lhs          # > 0 means violated
        viol_y = -y                          # > 0 means y < 0
        viol_z = -z                          # > 0 means z < 0
        residuals = np.maximum(0.0, np.maximum(viol_cone, np.maximum(viol_y, viol_z)))
        return residuals[0] if residuals.size == 1 else residuals

    def save_dual_value(self, value) -> None:
        n_x = self.args[0].shape[0]
        n_cones = self.args[1].size
        if isinstance(value, (list, tuple)):
            # recover_dual returns [dx_dual (n_cones, n_x), dy_dual (n_cones,), dz_dual (n_cones,)]
            dx = np.asarray(value[0])   # (n_cones, n_x)
            dy = np.asarray(value[1])   # (n_cones,)
            dz = np.asarray(value[2])   # (n_cones,)
        else:
            # Fallback: flat vector [x (n_x), y, z] — should not occur with recover_dual
            value = np.reshape(value, (-1, n_x + 2))
            dx = value[:, :n_x]
            dy = value[:, n_x]
            dz = value[:, n_x + 1]
        if n_cones == 1:
            # Scalar case: squeeze to match primal shapes
            self.dual_variables[0].save_value(dx[0])   # (n_x,)
            self.dual_variables[1].save_value(dy[0])   # scalar
            self.dual_variables[2].save_value(dz[0])   # scalar
        else:
            # Batched case: dx is (n_cones, n_x), transpose to (n_x, n_cones)
            self.dual_variables[0].save_value(dx.T)
            self.dual_variables[1].save_value(dy)
            self.dual_variables[2].save_value(dz)

    def get_data(self):
        return [self.axis, self.id]

    def is_dcp(self, dpp: bool = False) -> bool:
        """An RSOC constraint is DCP if all arguments are affine."""
        if dpp:
            with scopes.dpp_scope():
                return all(arg.is_affine() for arg in self.args)
        return all(arg.is_affine() for arg in self.args)

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    def _dual_cone(self, *args):
        """The dual cone of the RSOC is itself (self-dual)."""
        if not args:
            return RSOC(
                self.dual_variables[0],
                self.dual_variables[1],
                self.dual_variables[2],
                self.axis,
            )
        else:
            assert len(args) == 3
            return RSOC(args[0], args[1], args[2], self.axis)
