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

from typing import Any

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.concatenate import concatenate


def _max_ndim(arr):
    if isinstance(arr, list):
        return max(_max_ndim(a) for a in arr)
    return AffAtom.cast_to_const(arr).ndim


def _block_depth(arr):
    depth = 0
    while isinstance(arr, list):
        depth += 1
        arr = arr[0]
    return depth


def _block_rec(arr, level, depth, ndim):
    if not isinstance(arr, list):
        return AffAtom.cast_to_const(arr)

    sub = [_block_rec(a, level + 1, depth, ndim) for a in arr]

    axis = ndim - depth + level
    return concatenate(sub, axis=axis)


def block(arr: Any):
    if not isinstance(arr, list):
        raise ValueError("Input must be a nested list structure.")

    depth = _block_depth(arr)
    ndim = max(_max_ndim(arr), depth)

    def promote(x):
        if isinstance(x, list):
            return [promote(a) for a in x]
        expr = AffAtom.cast_to_const(x)
        if expr.ndim < ndim:
            new_shape = (1,) * (ndim - expr.ndim) + expr.shape
            return expr.reshape(new_shape, order="F")
        return expr

    promoted = promote(arr)

    return _block_rec(promoted, level=0, depth=depth, ndim=ndim)
