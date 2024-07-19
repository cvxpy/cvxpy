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
from functools import reduce
from operator import mul
from typing import List, Tuple

import numpy as np


def squeezed(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(dim for dim in shape if dim != 1)


def sum_shapes(shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    Give the shape resulting from summing a list of shapes.

    Summation semantics are exactly the same as NumPy's, including
    broadcasting.

    Parameters
    ----------
    shapes : list of tuples
        The shapes of the summands

    Returns
    -------
    tuple
        The shape of the sum.

    Raises
    ------
    ValueError
        If the shapes are not compatible.
    """
    return np.broadcast_shapes(*shapes)


def mul_shapes_promote(
    lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Promotes shapes as necessary and returns promoted shape of product.

    If lh_shape is of length one, prepend a one to it.
    If rh_shape is of length one, append a one to it.
    If either shape is greater than 2, apply broadcasting rules on
    the outer dimensions.
    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplication operation.
    rh_shape : tuple
        The right-hand shape of a multiplication operation.

    Returns
    -------
    tuple
        The promoted left-hand shape.
    tuple
        The promoted right-hand shape.
    tuple
        The promoted shape of the product.

    Raises
    ------
    ValueError
        If either of the shapes are scalars, or if the shapes are incompatible.
    """
    if not lh_shape or not rh_shape:
        raise ValueError("Multiplication by scalars is not permitted.")

    # Promote 1D shapes to 2D
    lh_shape = (1,) + lh_shape if len(lh_shape) == 1 else lh_shape
    rh_shape = rh_shape + (1,) if len(rh_shape) == 1 else rh_shape

    if lh_shape[-1] != rh_shape[-2]:
        raise ValueError("Incompatible dimensions %s %s" % (lh_shape, rh_shape))
    
    # Calculate resulting shape for higher-dimensional arrays
    if len(lh_shape) > 2 or len(rh_shape) > 2:
        try:
            outer_dims = np.broadcast_shapes(lh_shape[:-2], rh_shape[:-2])
        except ValueError:
            raise ValueError("Incompatible dimensions %s %s" % (lh_shape, rh_shape))
        shape = outer_dims + (lh_shape[-2], rh_shape[-1])
    else:
        shape = (lh_shape[-2], rh_shape[-1])
    return (lh_shape, rh_shape, shape)


def mul_shapes(lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Give the shape resulting from multiplying two shapes.

    Adheres the semantics of np.matmul and additionally permits multiplication
    by scalars. The broadcasting rules are the same as NumPy's.

    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplication operation.
    rh_shape :  tuple
        The right-hand shape of a multiplication operation.

    Returns
    -------
    tuple
        The shape of the product as per matmul semantics.

    Raises
    ------
    ValueError
        If either of the shapes are scalar.
    """
    lh_old, rh_old = lh_shape, rh_shape
    lh_shape, rh_shape, shape = mul_shapes_promote(lh_shape, rh_shape)
    if lh_shape != lh_old:
        shape = shape[:-2] + (shape[-1],)
    if rh_shape != rh_old:
        shape = shape[:-1]
    return shape


def size_from_shape(shape) -> int:
    """
    Compute the size of a given shape by multiplying the sizes of each axis.

    This is a replacement for np.prod(shape, dtype=int) which is much slower for
    small arrays than the implementation below.

    Parameters
    ----------
    shape : tuple
        a tuple of integers describing the shape of an object

    Returns
    -------
    int
        The size of an object corresponding to shape.
    """
    return reduce(mul, shape, 1)
