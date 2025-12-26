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
from numpy.lib.array_utils import normalize_axis_index
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
        axis_norm = normalize_axis_index(axis, ndim)

        if ndim == 2:
            m, n = shape
            if axis_norm == 0:
                # Sum over rows for each column
                # In Fortran order, column j uses indices j*m to (j+1)*m-1
                block_indices = [np.arange(j * m, (j + 1) * m) for j in range(n)]
            else:
                # Sum over columns for each row
                # In Fortran order, row i uses indices i, i+m, i+2m, ..., i+(n-1)*m
                block_indices = [np.arange(i, m * n, m) for i in range(m)]
        else:
            raise NotImplementedError(
                f"axis parameter only supported for 2D input, got {ndim}D."
            )

    if isinstance(affine_expr, Variable):
        return SymbolicQuadForm(affine_expr, quad_mat, expr, block_indices=block_indices), []
    else:
        t = Variable(affine_expr.shape)
        return SymbolicQuadForm(t, quad_mat, expr, block_indices=block_indices), [affine_expr == t]
