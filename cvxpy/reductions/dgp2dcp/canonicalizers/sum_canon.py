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

import math

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.reductions.dgp2dcp.canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.util import explicit_sum


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        summation = explicit_sum(X)
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape, order='F'), []

    axis = expr.axis
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = len(X.shape)

    # Permute so non-reduce axes come first, reduce axes come last
    keep = [i for i in range(ndim) if i not in axes]
    perm = keep + list(axes)
    X_perm = transpose(X, axes=perm) if perm != list(range(ndim)) else X

    # Reshape to 2D: (n_output, n_reduce)
    # Use F-order to match the final reshape back to expr.shape.
    n_output = math.prod(X.shape[i] for i in keep)
    n_reduce = math.prod(X.shape[i] for i in axes)
    X_2d = reshape(X_perm, (n_output, n_reduce), order='F')

    rows = []
    for i in range(n_output):
        summation = explicit_sum(X_2d[i])
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape, order='F'), []
