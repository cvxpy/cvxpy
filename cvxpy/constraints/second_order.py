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

import numpy as np

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes


class SOC(Constraint):
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
            t = t.flatten()
        super(SOC, self).__init__([t, X], constr_id)

    def __str__(self) -> str:
        return "SOC(%s, %s)" % (self.args[0], self.args[1])

    @property
    def residual(self):
        t = self.args[0].value
        X = self.args[1].value
        if t is None or X is None:
            return None
        if self.axis == 0:
            X = X.T
        norms = np.linalg.norm(X, ord=2, axis=1)
        zero_indices = np.where(X <= -t)[0]
        averaged_indices = np.where(X >= np.abs(t))[0]
        X_proj = np.array(X)
        t_proj = np.array(t)
        X_proj[zero_indices] = 0
        t_proj[zero_indices] = 0
        avg_coeff = 0.5 * (1 + t/norms)
        X_proj[averaged_indices] = avg_coeff * X[averaged_indices]
        t_proj[averaged_indices] = avg_coeff * t[averaged_indices]
        return np.linalg.norm(np.concatenate([X, t], axis=1) -
                              np.concatenate([X_proj, t_proj], axis=1),
                              ord=2, axis=1)

    def get_data(self):
        """Returns info needed to reconstruct the object besides the args.

        Returns
        -------
        list
        """
        return [self.axis]

    def num_cones(self):
        """The number of elementwise cones.
        """
        return self.args[0].size

    @property
    def size(self) -> int:
        """The number of entries in the combined cones.
        """
        cone_size = 1 + self.args[1].shape[self.axis]
        return cone_size * self.num_cones()

    def cone_sizes(self) -> List[int]:
        """The dimensions of the second-order cones.

        Returns
        -------
        list
            A list of the sizes of the elementwise cones.
        """
        cone_size = 1 + self.args[1].shape[self.axis]
        return [cone_size] * self.num_cones()

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
        cone_size = 1 + self.args[1].shape[self.axis]
        value = np.reshape(value, newshape=(-1, cone_size))
        t = value[:, 0]
        X = value[:, 1:]
        if self.axis == 0:
            X = X.T
        self.dual_variables[0].save_value(t)
        self.dual_variables[1].save_value(X)
