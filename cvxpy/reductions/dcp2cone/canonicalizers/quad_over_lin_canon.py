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
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from cvxpy.atoms.affine.broadcast_to import broadcast_to
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape as shape_utils
from cvxpy.utilities.solver_context import SolverInfo


def quad_over_lin_canon(expr, args, solver_context: SolverInfo | None = None):
    """Canonicalize quad_over_lin to SOC constraints.

    quad_over_lin(x, y) = ||x||_2^2 / y

    Equivalent SOC: ||[y-t, 2x]||_2 <= y+t
    Which gives: (y-t)^2 + 4||x||^2 <= (y+t)^2
                 => ||x||^2 <= t*y
    """
    x = args[0]
    y = args[1]
    broadcast_shape = shape_utils.sum_shapes([x.shape, y.shape])
    x = broadcast_to(x, broadcast_shape)
    if not y.is_scalar():
        y = broadcast_to(y, broadcast_shape)
    axis = expr.axis
    ndim = len(broadcast_shape)

    # Normalize axis/axes to tuple
    axes = None
    if axis is not None:
        if isinstance(axis, int):
            axes = (normalize_axis_index(axis, ndim),)
        else:
            axes = normalize_axis_tuple(axis, ndim)

    if not y.is_scalar():
        # Element-wise: x_i^2 / y_i via batched SOC
        n = x.size
        t = Variable(x.shape)  # full element-wise shape
        x_flat = x.flatten(order="F")
        y_flat = y.flatten(order="F")
        t_flat = t.flatten(order="F")
        # Each column j of X_soc is [y_j - t_j, 2*x_j]
        X_soc = vstack([
            reshape(y_flat - t_flat, (1, n), order="F"),
            reshape(2 * x_flat, (1, n), order="F"),
        ])
        constraints = [SOC(t=y_flat + t_flat, X=X_soc, axis=0)]
        return Sum(t, axis=axes, keepdims=expr.keepdims), constraints

    y = y.flatten(order="F")

    if axes is None:
        # Scalar output - single SOC constraint
        t = Variable(1,)
        constraints = [SOC(t=y + t, X=hstack([y - t, 2 * x.flatten(order="F")]), axis=0)]
        return t, constraints

    shape = x.shape
    # Axis specified - use vectorized batched SOC
    axes_set = set(axes)
    output_dims = [i for i in range(ndim) if i not in axes_set]
    reduce_dims = sorted(axes)

    output_shape = tuple(shape[i] for i in output_dims)
    reduce_shape = tuple(shape[i] for i in reduce_dims)
    n_outputs = int(np.prod(output_shape)) if output_shape else 1
    reduce_size = int(np.prod(reduce_shape))

    # Create output variable with the correct shape
    t = Variable(expr.shape)
    t_flat = t.flatten(order="F")

    # Permute dimensions: reduce_dims first, then output_dims
    # This ensures reduced elements are contiguous in Fortran order
    perm = list(reduce_dims) + output_dims
    if perm != list(range(ndim)):
        x_perm = transpose(x, axes=perm)
    else:
        x_perm = x

    # Reshape to 2D: (reduce_size, n_outputs)
    x_2d = reshape(x_perm, (reduce_size, n_outputs), order="F")

    # Build vectorized SOC constraint
    # For each output j: ||[y-t[j], 2*x_col_j]||_2 <= y+t[j]
    # X_soc has shape (1 + reduce_size, n_outputs), columns are cones
    y_minus_t_row = reshape(y - t_flat, (1, n_outputs), order="F")
    X_soc = vstack([y_minus_t_row, 2 * x_2d])
    t_soc = y + t_flat

    return t, [SOC(t=t_soc, X=X_soc, axis=0)]
