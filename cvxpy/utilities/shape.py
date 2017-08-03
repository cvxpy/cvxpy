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
    """Removes single-dimensional entries from a shape.
    """
    return tuple([dim for dim in shape if dim != 1])


def sum_shapes(shapes):
    """Give the shape resulting from summing a list of shapes.

    Summation semantics are exactly the same as NumPy's, including
    broadcasting.

    Args:
        shapes: A list of shape tuples.

    Returns:
        The shape of the sum.

    Raises:
        ValueError if the shapes are not compatible.
    """
    shape = shapes[0]
    for t in shapes[1:]:
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
        shape = prefix + suffix
    return tuple(shape)


def mul_shapes(lh_shape, rh_shape):
    """Give the shape resulting from multiplying two shapes.

    Adheres the semantics of np.matmul but permits multiplication by
    scalars, where a scalar is defined as either a 0D or an all ones shape.

    Args:
        lh_shape: A shape tuple.
        rh_shape: A shape tuple.

    Returns:
        The shape of the product.
    """
    # Multiplication by a scalar (a 0D or all ones shape) is always allowed.
    if not squeezed(lh_shape):
        return rh_shape
    elif not squeezed(rh_shape):
        return lh_shape
    else:
        lhs_mat_shape = lh_shape[-2:]
        rhs_mat_shape = rh_shape[-2:]
        if lhs_mat_shape[1] != rhs_mat_shape[0]:
            raise ValueError("Incompatible dimensions %s %s" % (
                lh_shape, rh_shape))
        if lh_shape[:-2] != rh_shape[:-2]:
            raise ValueError("Incompatible dimensions %s %s" % (
                lh_shape, rh_shape))
        return tuple(list(lh_shape[:-2]) + [lhs_mat_shape[0]] +
                     [rhs_mat_shape[1]])
