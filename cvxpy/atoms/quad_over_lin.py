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
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

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
        self._broadcast_shape = u.shape.sum_shapes([self.args[0].shape, self.args[1].shape])
        if self.axis is not None:
            ndim = len(self._broadcast_shape)
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

        squared = np.broadcast_to(squared, self._broadcast_shape)

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
        if self.axis is not None:
            return [None, None]

        X = values[0]
        y = values[1]
        if np.any(y <= 0):
            return [None, None]

        if self.args[0].is_complex():
            X_squared = np.square(X.real) + np.square(X.imag)
        else:
            X_squared = np.square(X)

        if self.args[1].is_scalar():
            # Scalar y: sum before dividing to avoid broadcasting overhead.
            DX = 2.0 * X / y
            DX = np.reshape(DX, (self.args[0].size, 1), order='F')
            DX = scipy.sparse.csc_array(DX)
            Dy = sp.csc_array([[(-X_squared.sum() / np.square(y)).item()]])
            return [DX, Dy]

        bc_shape = self._broadcast_shape
        X_bc = np.broadcast_to(X, bc_shape)
        X_sq_bc = np.broadcast_to(X_squared, bc_shape)
        y_bc = np.broadcast_to(y, bc_shape)

        DX = _unbroadcast(2.0 * X_bc / y_bc, X.shape)
        DX = scipy.sparse.csc_array(
            np.reshape(DX, (self.args[0].size, 1), order='F')
        )

        Dy = _unbroadcast(-X_sq_bc / np.square(y_bc), y.shape)
        Dy = sp.csc_array(
            np.reshape(Dy, (self.args[1].size, 1), order='F')
        )

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
        if self.args[1].is_complex():
            raise ValueError(
                "The second argument to quad_over_lin cannot be complex."
            )

        # Check if the shapes are broadcast-compatible
        try:
            bshape = u.shape.sum_shapes([self.args[0].shape, self.args[1].shape])
        except ValueError:
            raise ValueError(
                "The shapes of the arguments to quad_over_lin are incompatible."
            )
        # Validate axis against broadcast ndim
        if self.axis is not None:
            axes = [self.axis] if isinstance(self.axis, int) else self.axis
            normalize_axis_tuple(axes, len(bshape))
        Atom.validate_arguments(self)

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        shape = list(u.shape.sum_shapes([self.args[0].shape, self.args[1].shape]))
        ndim = len(shape)
        if self.axis is None:
            return (1,) * len(shape) if self.keepdims else ()
        elif isinstance(self.axis, int):
            # Normalize negative axis
            axis = normalize_axis_index(self.axis, ndim)
            if self.keepdims:
                shape[axis] = 1
            else:
                shape = shape[:axis] + shape[axis+1:]
        else:
            # Normalize each axis in the list
            axes =  normalize_axis_tuple(self.axis, ndim)
            if self.keepdims:
                for axis in axes:
                    shape[axis] = 1
            else:
                shape[:] = [shape[i] for i in range(len(shape)) if i not in axes]
        return tuple(shape)

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

def _unbroadcast(
    arr: np.ndarray,
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    """Sum an array from its broadcast shape back to target_shape.

    target_shape must broadcast to arr.shape.
    """
    try:
        result_shape = np.broadcast_shapes(target_shape, arr.shape)
    except ValueError:
        result_shape = None
    if result_shape != arr.shape:
        raise ValueError(
            f"target_shape {target_shape} does not broadcast to arr.shape {arr.shape}."
        )
    # Undo left padding.
    pad = len(arr.shape) - len(target_shape)
    reduce_axes = list(range(pad))
    for i, (a, b) in enumerate(zip(target_shape, arr.shape[pad:])):
        if a == 1 and b > 1:
            reduce_axes.append(i + pad)
    if reduce_axes:
        # Sum over the reduce axes to get the target shape, and re-add the size-1 axes.
        return arr.sum(axis=tuple(reduce_axes)).reshape(target_shape)
    return arr
