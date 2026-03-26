"""
Copyright 2017 Robin Verschueren

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
from scipy.sparse import diags, eye_array

from cvxpy.atoms.affine.broadcast_to import broadcast_to
from cvxpy.atoms.affine.diag import diag_vec
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape as shape_utils
from cvxpy.utilities.solver_context import SolverInfo


def quad_over_lin_canon(expr, args, solver_context: SolverInfo | None = None):
    broadcast_shape = shape_utils.sum_shapes([args[0].shape, args[1].shape])
    affine_expr = broadcast_to(args[0], broadcast_shape)
    y = args[1]
    if not y.is_scalar():
        y = broadcast_to(y, broadcast_shape)
    axis = expr.axis

    if len(y.parameters()) > 0:
        # TODO both codepaths produce an intermediate dense matrix.
        # but it should be sparse the whole time.
        if y.is_scalar():
            quad_mat = eye_array(affine_expr.size) / y
        else:
            quad_mat = diag_vec(1.0 / y.flatten(order="F"))
    else:
        if y.is_scalar():
            quad_mat = eye_array(affine_expr.size) / y.value
        else:
            y_flat = np.asarray(y.value).ravel(order="F")
            quad_mat = diags(1.0 / y_flat).tocsc()

    # Compute block_indices if axis is specified
    block_indices = None
    if axis is not None:
        shape = affine_expr.shape
        ndim = len(shape)

        # Normalize axis/axes to tuple
        if isinstance(axis, int):
            axes = (normalize_axis_index(axis, ndim),)
        else:
            axes = normalize_axis_tuple(axis, ndim)
            
        block_indices = _compute_block_indices(shape, axes)

    if isinstance(affine_expr, Variable):
        return SymbolicQuadForm(affine_expr, quad_mat, expr, block_indices=block_indices), []
    else:
        t = Variable(affine_expr.shape)
        return SymbolicQuadForm(t, quad_mat, expr, block_indices=block_indices), [affine_expr == t]


def _compute_block_indices(shape, axes):
    """Compute block indices for reducing along specified axes (Fortran order).

    Parameters
    ----------
    shape : tuple
        Shape of input array
    axes : tuple of int
        Axes being reduced (must be normalized/positive)

    Returns
    -------
    list of np.ndarray
        block_indices[j] = array of input indices for output element j
    """
    ndim = len(shape)
    axes_set = set(axes)

    # Output shape (dimensions not in axes)
    output_dims = [i for i in range(ndim) if i not in axes_set]
    output_shape = tuple(shape[i] for i in output_dims)
    n_outputs = int(np.prod(output_shape)) if output_shape else 1

    # Dimensions being reduced
    reduce_shape = tuple(shape[i] for i in sorted(axes))
    reduce_size = int(np.prod(reduce_shape))

    # Vectorized computation of all indices at once
    # out_multi: tuple of arrays, each shape (n_outputs,)
    if output_shape:
        out_multi = np.unravel_index(np.arange(n_outputs), output_shape, order="F")
    else:
        out_multi = ()

    # reduce_multi: tuple of arrays, each shape (reduce_size,)
    if reduce_shape:
        reduce_multi = np.unravel_index(np.arange(reduce_size), reduce_shape, order="F")
    else:
        reduce_multi = ()

    # Build input multi-index arrays with broadcasting
    # Each element will have shape (n_outputs, reduce_size)
    input_multi = []
    out_ptr = 0
    reduce_ptr = 0
    for i in range(ndim):
        if i in axes_set:
            # Broadcast reduce indices: (reduce_size,) -> (n_outputs, reduce_size)
            input_multi.append(np.broadcast_to(reduce_multi[reduce_ptr], (n_outputs, reduce_size)))
            reduce_ptr += 1
        else:
            # Broadcast output indices: (n_outputs,) -> (n_outputs, reduce_size)
            input_multi.append(
                np.broadcast_to(out_multi[out_ptr][:, np.newaxis], (n_outputs, reduce_size))
            )
            out_ptr += 1

    # Compute all flat indices at once: shape (n_outputs, reduce_size)
    flat_indices_2d = np.ravel_multi_index(input_multi, shape, order="F")

    # Return as list of arrays for compatibility with SymbolicQuadForm
    return [flat_indices_2d[j] for j in range(n_outputs)]
