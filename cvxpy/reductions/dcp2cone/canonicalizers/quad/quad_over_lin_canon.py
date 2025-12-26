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
from scipy.sparse import eye_array

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variable import Variable


def quad_over_lin_canon(expr, args):
    affine_expr = args[0]
    y = args[1]
    axis = expr.axis

    # Simplify if y has no parameters.
    if len(y.parameters()) == 0:
        quad_mat = eye_array(affine_expr.size) / y.value
    else:
        # TODO this codepath produces an intermediate dense matrix.
        # but it should be sparse the whole time.
        quad_mat = eye_array(affine_expr.size) / y

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

    block_indices = []

    for out_idx in range(n_outputs):
        # Convert flat output index to multi-index in output shape
        if output_shape:
            out_multi = np.unravel_index(out_idx, output_shape, order='F')
        else:
            out_multi = ()

        indices = []
        for reduce_idx in range(reduce_size):
            # Convert reduce_idx to multi-index in reduce_shape
            if reduce_shape:
                reduce_multi = np.unravel_index(reduce_idx, reduce_shape, order='F')
            else:
                reduce_multi = ()

            # Build full input multi-index by interleaving
            input_multi = [0] * ndim
            out_ptr = 0
            reduce_ptr = 0
            for i in range(ndim):
                if i in axes_set:
                    input_multi[i] = reduce_multi[reduce_ptr]
                    reduce_ptr += 1
                else:
                    input_multi[i] = out_multi[out_ptr]
                    out_ptr += 1

            # Convert to flat index
            flat_idx = np.ravel_multi_index(input_multi, shape, order='F')
            indices.append(flat_idx)

        block_indices.append(np.array(indices))

    return block_indices
