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

# Utility functions to handle indexing/slicing into an expression.

import numpy as np

def validate_key(key, shape):
    """Check if the key is a valid index.

    Args:
        key: The key used to index/slice.
        shape: The shape of the expression.

    Returns:
        The key as a tuple of slices.

    Raises:
        Error: Index/slice out of bounds.
    """
    rows, cols = shape.size
    # Change single indices for vectors into double indices.
    if not isinstance(key, tuple):
        if rows == 1:
            key = (slice(0, 1, None), key)
        elif cols == 1:
            key = (key, slice(0, 1, None))
        else:
            raise IndexError("Invalid index/slice.")
    # Change numbers into slices and ensure all slices have a start and step.
    key = (format_slice(slc, dim) for slc, dim in zip(key, shape.size))
    return tuple(key)

def format_slice(key_val, dim):
    """Converts part of a key into a slice with a start and step.

    Uses the same syntax as numpy.

    Args:
        key_val: The value to convert into a slice.
        dim: The length of the dimension being sliced.

    Returns:
        A slice with a start and step.
    """
    if isinstance(key_val, slice):
        return key_val
    else:
        key_val = wrap_neg_index(key_val, dim)
        if 0 <= key_val < dim:
            return slice(key_val, key_val + 1, 1)
        else:
            raise IndexError("Index/slice out of bounds.")

def wrap_neg_index(index, dim):
    """Converts a negative index into a positive index.

    Args:
        index: The index to convert. Can be None.
        dim: The length of the dimension being indexed.
    """
    if index is not None and index < 0:
        index %= dim
    return index

def index_to_slice(idx):
    """Converts an index to a slice.

    Args:
        idx: int
            The index.

    Returns:
    slice
        A slice equivalent to the index.
    """
    return slice(idx, idx+1, None)

def slice_to_str(slc):
    """Converts a slice into a string.
    """
    if is_single_index(slc):
        return str(slc.start)
    endpoints = [none_to_empty(val) for val in (slc.start, slc.stop)]
    if slc.step is not None and slc.step != 1:
        return "%s:%s:%s" % (endpoints[0], endpoints[1], slc.step)
    else:
        return "%s:%s" % (endpoints[0], endpoints[1])

def none_to_empty(val):
    """Converts None to an empty string.
    """
    if val is None:
        return ''
    else:
        return val

def is_single_index(slc):
    """Is the slice equivalent to a single index?
    """
    if slc.step is None:
        step = 1
    else:
        step = slc.step
    return slc.start is not None and \
           slc.stop is not None and \
           slc.start + step >= slc.stop

def size(key, shape):
    """Finds the dimensions of a sliced expression.

    Args:
        key: The key used to index/slice.
        shape: The shape of the expression.

    Returns:
        The dimensions of the expression as (rows, cols).
    """
    dims = []
    for i in range(2):
        selection = np.arange(shape.size[i])[key[i]]
        size = np.size(selection)
        dims.append(size)
    return tuple(dims)

def to_str(key):
    """Converts a key (i.e. two slices) into a string.
    """
    return (slice_to_str(key[0]), slice_to_str(key[1]))
