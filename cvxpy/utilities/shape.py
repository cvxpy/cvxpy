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


def sum_shapes(shapes):
    """Give the shape resulting from summing a list of shapes.

    Args:
        shapes: A list of (row, col) tuples.

    Returns:
        The shape (row, col) shape of the sum.
    """
    rows = max([shape[0] for shape in shapes])
    cols = max([shape[1] for shape in shapes])
    # Validate shapes.
    for shape in shapes:
        if not shape == (1, 1) and shape != (rows, cols):
            raise ValueError(
                "Incompatible dimensions" + len(shapes)*" %s" % tuple(shapes))
    return (rows, cols)


def mul_shapes(lh_shape, rh_shape):
    """Give the shape resulting from multiplying two shapes.

    Args:
        lh_shape: A (row, col) tuple.
        rh_shape: A (row, col) tuple.

    Returns:
        The shape (row, col) shape of the product.
    """
    if lh_shape == (1, 1):
        return rh_shape
    elif rh_shape == (1, 1):
        return lh_shape
    else:
        if lh_shape[1] != rh_shape[0]:
            raise ValueError("Incompatible dimensions %s %s" % (
                lh_shape, rh_shape))
        return (lh_shape[0], rh_shape[1])
