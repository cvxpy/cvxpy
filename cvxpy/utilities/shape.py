"""
Copyright 2017 Steven Diamond

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
