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
        if args is None:
            return SOC(self.dual_variables[0], self.dual_variables[1], self.axis)
        else:
            # some assertions for verifying `args`
            args_shapes = [arg.shape for arg in args]
            instance_args_shapes = [arg.shape for arg in self.args]
            assert len(args) == len(self.args)
            assert args_shapes == instance_args_shapes
            return SOC(args[0], args[1], self.axis)


class RSOC(Cone):
    """Rotated second-order cone constraint.

    Constrains (x, y, z) to satisfy::

        y * z >= ||x||^2
        y >= 0, z >= 0

    Parameters
    ----------
    x : Expression
        The "vector" part of the RSOC.
    y : Expression
        First scalar part of the RSOC.
    z : Expression
        Second scalar part of the RSOC.
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

    def num_cones(self):
        return 1

    def cone_sizes(self) -> List[int]:
        x = self.args[0]
        return [1 + x.size]

    @property
    def size(self) -> int:
        return self.cone_sizes()[0]

    @property
    def residual(self) -> Optional[float]:
        """Residual: distance to the nearest point in the RSOC.

        Computed via the equivalent SOC form SOC(y+z, [2*x; y-z]).
        """
        x, y, z = self.args

        if any(arg.value is None for arg in self.args):
            return None

        xv = np.atleast_1d(np.array(x.value, dtype=float)).flatten()
        yv = float(y.value)
        zv = float(z.value)

        # Reuse the SOC residual logic on raw numpy values to avoid shape
        # issues when building CVXPY expressions for 1-D / scalar args.
        t = np.array([yv + zv])
        X = np.atleast_2d(np.concatenate([2 * xv, [yv - zv]]))  # (1, n+1)

        t_proj = np.zeros(t.shape)
        X_proj = np.zeros(X.shape)
        norms = np.linalg.norm(X, ord=2, axis=1)

        t_geq = t >= norms
        t_proj[t_geq] = t[t_geq]
        X_proj[t_geq] = X[t_geq]

        abs_t_less = np.abs(t) < norms
        avg_coeff = 0.5 * (1 + t / norms)
        X_proj[abs_t_less] = avg_coeff[abs_t_less, None] * X[abs_t_less]
        t_proj[abs_t_less] = avg_coeff[abs_t_less] * norms[abs_t_less]

        Xt = np.concatenate([X, t[:, None]], axis=1)
        Xt_proj = np.concatenate([X_proj, t_proj[:, None]], axis=1)
        return float(np.linalg.norm(Xt - Xt_proj, ord=2, axis=1)[0])

    def save_dual_value(self, value) -> None:
        """Recover RSOC dual variables from the lowered SOC dual vector.

        When RSOC(x, y, z) is lowered to SOC(y+z, [2*x, y-z]), the solver
        returns the dual vector [lam_t, lam_x..., lam_yz] for that SOC.
        The RSOC dual components are recovered as:

            dual for x : 2 * lam_x
            dual for y : lam_t + lam_yz
            dual for z : lam_t - lam_yz
        """
        value = np.atleast_1d(np.array(value, dtype=float)).flatten()
        n = self.args[0].size
        lam_t = value[0]
        lam_x = value[1:n + 1]
        lam_yz = value[n + 1]
        self.dual_variables[0].save_value(
            np.reshape(2 * lam_x, self.args[0].shape, order='F'))
        self.dual_variables[1].save_value(float(lam_t + lam_yz))
        self.dual_variables[2].save_value(float(lam_t - lam_yz))

    def _dual_cone(self, *args):
        """Returns a constraint representing the dual cone of RSOC.

        The dual cone of K = {(x, y, z) : y*z >= ||x||^2, y >= 0, z >= 0}
        is K* = {(mu_x, mu_y, mu_z) : 4*mu_y*mu_z >= ||mu_x||^2},
        which is equivalent to RSOC(mu_x / 2, mu_y, mu_z).
        """
        if not args:
            x_d = self.dual_variables[0]
            y_d = self.dual_variables[1]
            z_d = self.dual_variables[2]
        else:
            assert len(args) == len(self.args)
            x_d, y_d, z_d = args[0], args[1], args[2]
        return RSOC(x_d / 2, y_d, z_d)
