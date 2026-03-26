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

from typing import List, Tuple

import numpy as np
import scipy as scipy
import scipy.sparse as sp

import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom, normalize_axis
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants.parameter import is_param_free


class quad_over_lin(AxisAtom):
    """:math:`(sum_{ij}X^2_{ij})/y`

    Computes :math:`X^2_{ij}/y_{ij}` (or :math:`X^2_{ij}/y` for scalar y),
    summed over all elements by default (axis=None).

    Use ``axis=()`` for element-wise output with no reduction.
    """
    _allow_complex = True

    def __init__(
        self,
        x,
        y,
        axis: None | int | tuple[int, ...] = None,
        keepdims: bool = False
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        # Call Atom.__init__ directly since we have two args
        Atom.__init__(self, x, y)
        # Normalize axis after init so self.args is available.
        if self.axis is not None:
            ndim = len(self.args[0].shape)
            self.axis = normalize_axis(self.axis, ndim)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        x_val = values[0]
        y_val = values[1]
        if self.args[0].is_complex():
            squared = np.square(x_val.imag) + np.square(x_val.real)
        else:
            squared = np.square(x_val)

        if self.args[1].is_scalar():
            return squared.sum(axis=self.axis, keepdims=self.keepdims) / y_val.item()
        else:
            result = squared / y_val
            return result.sum(axis=self.axis, keepdims=self.keepdims)

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        # y > 0.
        return [self.args[1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Gradient only implemented for full reduction (axis=None).
        if self.axis is not None:
            return [None, None]

        X = values[0]
        y = values[1]
        if np.any(y <= 0):
            return [None, None]

        y_squared = np.square(y)
        if self.args[0].is_complex():
            X_squared = np.square(X.real) + np.square(X.imag)
        else:
            X_squared = np.square(X)

        # DX: (n, 1), Dy: (1, 1).
        DX = 2.0 * X / y
        DX = np.reshape(DX, (self.args[0].size, 1), order='F')
        DX = scipy.sparse.csc_array(DX)
        if self.args[1].is_scalar():
            Dy = (-X_squared.sum() / y_squared).item()
        else:
            Dy = (-(X_squared / y_squared).sum()).item()
        Dy = sp.csc_array([[Dy]])

        return [DX, Dy]

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        # Disable DPP when the second argument is a parameter.
        if u.scopes.dpp_scope_active():
            return is_param_free(self.args[1])
        else:
            return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_smooth(self) -> bool:
        """Is the atom smooth?"""
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return (idx == 0) and self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return ((idx == 0) and self.args[idx].is_nonpos()) or (idx == 1)

    def validate_arguments(self) -> None:
        """Check dimensions of arguments.
        """
        if not (self.args[1].is_scalar()
                or self.args[1].shape == self.args[0].shape):
            raise ValueError(
                "The second argument to quad_over_lin must be a scalar "
                "or have the same shape as the first argument."
            )
        if self.args[1].is_complex():
            raise ValueError(
                "The second argument to quad_over_lin cannot be complex."
            )
        # AxisAtom.validate_arguments handles axis validation
        super(quad_over_lin, self).validate_arguments()

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine and y is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        """A quadratic term if y is constant.
        """
        return self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        """Quadratic of piecewise affine if x is PWL and y is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()
