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
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from numpy.lib.array_utils import normalize_axis_index

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.expressions.expression import Expression


class cumsum(AffAtom, AxisAtom):
    """
    Cumulative sum of the elements of an expression.

    Attributes
    ----------
    expr : CVXPY expression
        The expression being summed.
    axis : int, optional
        The axis to sum across. If None, the array is flattened before cumsum.
        Note: NumPy's default is axis=None, while CVXPY defaults to axis=0.
    """
    def __init__(self, expr: Expression, axis: Optional[int] = 0) -> None:
        super(cumsum, self).__init__(expr, axis)

    def validate_arguments(self) -> None:
        """Validate axis, but handle 0D arrays specially."""
        if self.args[0].ndim == 0:
            if self.axis is not None:
                warnings.warn(
                    "cumsum on 0-dimensional arrays currently returns a scalar, "
                    "but in a future CVXPY version it will return a 1-element "
                    "array to match numpy.cumsum behavior. Additionally, only "
                    "axis=0, axis=-1, or axis=None will be valid for 0D arrays.",
                    FutureWarning
                )
        else:
            super().validate_arguments()

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """
        Returns the cumulative sum of elements of an expression over an axis.
        """
        return np.cumsum(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
        """Flattened if axis=None, otherwise same as input."""
        if self.axis is None:
            return (self.args[0].size,)
        return self.args[0].shape

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        ndim = len(values[0].shape)
        axis = self.axis

        # Handle axis=None: treat as 1D cumsum over C-order flattened array
        if axis is None:
            dim = values[0].size
            # Lower triangular matrix = cumsum gradient in C-order space
            tril = sp.csc_array(np.tril(np.ones((dim, dim))))
            # Permutation to convert F-order vectorized input to C-order
            # P[i, j] = 1 means output[i] = input[j], so P @ f_vec = c_vec
            c_order_indices = np.arange(dim).reshape(values[0].shape, order='F').flatten(order='C')
            P = sp.csc_array((np.ones(dim), (np.arange(dim), c_order_indices)), shape=(dim, dim))
            # Gradient: input (F-order) -> C-order via P -> cumsum via tril -> output (1D)
            # dy = tril @ P @ dx_f, so gradient is tril @ P
            grad = tril @ P
            return [sp.csc_array(grad)]

        axis = normalize_axis_index(axis, ndim)
        dim = values[0].shape[axis]

        # Lower triangular matrix = cumsum gradient
        tril = sp.csc_array(np.tril(np.ones((dim, dim))))

        # Kronecker product: I_post ⊗ tril ⊗ I_pre
        # This works for all dimensions including 1D and 2D
        pre_size = int(np.prod(values[0].shape[:axis])) if axis > 0 else 1
        post_size = int(np.prod(values[0].shape[axis+1:])) if axis < ndim - 1 else 1

        grad = sp.kron(sp.kron(sp.eye_array(post_size), tril), sp.eye_array(pre_size))
        return [sp.csc_array(grad)]

    def get_data(self):
        """Returns the axis being summed."""
        return [self.axis]
