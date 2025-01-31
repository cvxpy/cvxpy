"""
Copyright 2024 the CVXPY developers

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

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.utilities.shape import mul_shapes_promote


def mulexpression_canon(expr, args):
    lhs = args[0]
    rhs = args[1]
    lhs_shape, rhs_shape, _ = mul_shapes_promote(lhs.shape, rhs.shape)
    lhs = reshape(lhs, lhs_shape, order='F')
    rhs = reshape(rhs, rhs_shape, order='F')
    rows = []
    # TODO(akshayka): Parallelize this for large matrices.
    for i in range(lhs.shape[0]):
        row = []
        for j in range(rhs.shape[1]):
            arr = hstack([lhs[i, k] + rhs[k, j] for k in range(lhs.shape[1])])
            row.append(log_sum_exp(arr))
        rows.append(row)
    mat = bmat(rows)
    if mat.shape != expr.shape:
        mat = reshape(mat, expr.shape, order='F')
    return mat, []
