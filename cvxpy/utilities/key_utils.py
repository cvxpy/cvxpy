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

# Utility functions to handle indexing/slicing into an expression.

import numpy as np
import numbers


def validate_key(key, shape):
    """Check if the key is a valid index.

    Args:
        key: The key used to index/slice.
        shape: The shape (rows, cols) of the expression.

    Returns:
        The key as a tuple of slices.

    Raises:
        Error: Index/slice out of bounds.
    """
    rows, cols = shape
    # Change single indices for vectors into double indices.
    if not isinstance(key, tuple):
        if rows == 1:
            key = (slice(0, 1, None), key)
        elif cols == 1:
            key = (key, slice(0, 1, None))
        else:
            raise IndexError("Invalid index/slice.")
    # Change numbers into slices and ensure all slices have a start and step.
    key = (format_slice(slc, dim) for slc, dim in zip(key, shape))
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
        key_val = slice(to_int(key_val.start),
                        to_int(key_val.stop),
                        to_int(key_val.step))
        return key_val
    else:
        # Convert to int.
        key_val = to_int(key_val)
        key_val = wrap_neg_index(key_val, dim)
        if 0 <= key_val < dim:
            return slice(key_val, key_val + 1, 1)
        else:
            raise IndexError("Index/slice out of bounds.")


def to_int(val):
    """Convert everything but None to an int.
    """
    if val is None:
        return val
    else:
        return int(val)


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
        shape: The shape (row, col) of the expression.

    Returns:
        The dimensions of the expression as (rows, cols).
    """
    dims = []
    for i in range(2):
        selection = np.arange(shape[i])[key[i]]
        size = np.size(selection)
        dims.append(size)
    return tuple(dims)


def to_str(key):
    """Converts a key (i.e. two slices) into a string.
    """
    return (slice_to_str(key[0]), slice_to_str(key[1]))


def is_special_slice(key):
    """Does the key contain a list, ndarray, or logical ndarray?
    """
    # Key is either a tuple of row, column keys or a single row key.
    if isinstance(key, tuple):
        if len(key) > 2:
            raise IndexError("Invalid index/slice.")
        key_elems = [key[0], key[1]]
    else:
        key_elems = [key]

    # Slices and int-like numbers are fine.
    for elem in key_elems:
        if not (isinstance(elem, (numbers.Number, slice)) or np.isscalar(elem)):
            return True

    return False
