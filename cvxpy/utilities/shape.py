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


def squeezed(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(dim for dim in shape if dim != 1)


def sum_shapes(shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Give the shape resulting from summing a list of shapes.

    Summation semantics are exactly the same as NumPy's, including
    broadcasting.

    Parameters
    ----------
    shapes : list of tuple
        The shapes to sum.

    Returns
    -------
    tuple
        The shape of the sum.

    Raises
    ------
    ValueError
        If the shapes are not compatible.
    """
    shape = shapes[0]
    for t in shapes[1:]:
        # Only allow broadcasting for 0D arrays or summation of scalars.
        if shape != t and len(squeezed(shape)) != 0 and len(squeezed(t)) != 0:
            raise ValueError(
                "Cannot broadcast dimensions " +
                len(shapes)*" %s" % tuple(shapes))

        longer = shape if len(shape) >= len(t) else t
        shorter = shape if len(shape) < len(t) else t
        offset = len(longer) - len(shorter)
        prefix = list(longer[:offset])
        suffix = []
        for d1, d2 in zip(reversed(longer[offset:]), reversed(shorter)):
            if d1 != d2 and not (d1 == 1 or d2 == 1):
                raise ValueError(
                    "Incompatible dimensions" +
                    len(shapes)*" %s" % tuple(shapes))
            new_dim = d1 if d1 >= d2 else d2
            suffix = [new_dim] + suffix
        shape = tuple(prefix + suffix)
    return tuple(shape)


def mul_shapes_promote(
    lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Promotes shapes as necessary and returns promoted shape of product.

    If lh_shape is of length one, prepend a one to it.
    If rh_shape is of length one, append a one to it.

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
        If either of the shapes are 0D.
    """
    if not lh_shape or not rh_shape:
        raise ValueError("Multiplication by scalars is not permitted.")

    if len(lh_shape) == 1:
        lh_shape = (1,) + lh_shape
    if len(rh_shape) == 1:
        rh_shape = rh_shape + (1,)

    lh_mat_shape = lh_shape[-2:]
    rh_mat_shape = rh_shape[-2:]
    if lh_mat_shape[1] != rh_mat_shape[0]:
        raise ValueError("Incompatible dimensions %s %s" % (
            lh_shape, rh_shape))
    if lh_shape[:-2] != rh_shape[:-2]:
        raise ValueError("Incompatible dimensions %s %s" % (
            lh_shape, rh_shape))
    return (lh_shape, rh_shape,
            tuple(list(lh_shape[:-2]) + [lh_mat_shape[0]] + [rh_mat_shape[1]]))


def mul_shapes(lh_shape: Tuple[int, ...], rh_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Give the shape resulting from multiplying two shapes.

    Adheres the semantics of np.matmul and additionally permits multiplication
    by scalars.

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
    lh_old = lh_shape
    rh_old = rh_shape
    lh_shape, rh_shape, shape = mul_shapes_promote(lh_shape, rh_shape)
    if lh_shape != lh_old:
        shape = shape[1:]
    if rh_shape != rh_old:
        shape = shape[:-1]
    return shape
