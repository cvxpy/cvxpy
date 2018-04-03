"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""


def squeezed(shape):
    return tuple(dim for dim in shape if dim != 1)


def sum_shapes(shapes):
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


def mul_shapes_promote(lh_shape, rh_shape):
    """Promotes shapes as necessary and returns promoted shape of product.

    If lh_shape is of length one, prepend a one to it.
    If rh_shape is of length one, append a one to it.

    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplciation operation.
    rh_shape : tuple
        The right-hand shape of a multiplciation operation.

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


def mul_shapes(lh_shape, rh_shape):
    """Give the shape resulting from multiplying two shapes.

    Adheres the semantics of np.matmul and additionally permits multiplication
    by scalars.

    Parameters
    ----------
    lh_shape : tuple
        The left-hand shape of a multiplciation operation.
    rh_shape :  tuple
        The right-hand shape of a multiplciation operation.

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
