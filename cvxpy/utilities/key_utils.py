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

# Utility functions to handle indexing/slicing into an expression.

from __future__ import division

import numbers
from typing import Optional, Tuple

import numpy as np


# TODO(akshayka): This module needs to be updated in order to handle
# NumPy 0/1D arrays.
def validate_key(key, shape: Tuple[int, ...]):
    """Check if the key is a valid index.

    Args:
        key: The key used to index/slice.
        shape: The shape (rows, cols) of the expression.

    Returns:
        The key as a tuple of slices.

    Raises:
        Error: Index/slice out of bounds.
    """
    key = to_tuple(key)
    if any(isinstance(k, float) or (isinstance(k, slice) and (
            isinstance(k.start, float) or
            isinstance(k.stop, float) or
            isinstance(k.step, float))) for k in key):
        raise IndexError("float is an invalid index type.")
    if len(key) == 0:
        raise IndexError("An index cannot be empty.")
    # Change single indices for vectors into double indices.
    none_count = sum(1 for elem in key if elem is None)
    slices = len(key) - none_count
    if slices > len(shape):
        raise IndexError("Too many indices for expression.")
    elif slices < len(shape):
        # Add : to the right.
        key = tuple(list(key) + [slice(None, None, None)]*(len(shape) - slices))
    # Change numbers into slices and ensure all slices have a start and step.
    return tuple(format_slice(slc, dim, i) for slc, dim, i in zip(key, shape, range(len(shape))))


def to_tuple(key):
    """Convert key to tuple if necessary.
    """
    if isinstance(key, tuple):
        return key
    else:
        return (key,)


def format_slice(key_val, dim, axis) -> Optional[slice]:
    """Converts part of a key into a slice with a start and step.

    Uses the same syntax as numpy.

    Args:
        key_val: The value to convert into a slice.
        dim: The length of the dimension being sliced.

    Returns:
        A slice with a start and step.
    """
    if key_val is None:
        return None
    elif isinstance(key_val, slice):
        step = to_int(key_val.step, 1)
        if step == 0:
            raise ValueError("step length cannot be 0")
        elif step > 0:
            start = np.clip(wrap_neg_index(to_int(key_val.start, 0), dim), 0, dim)
            stop = np.clip(wrap_neg_index(to_int(key_val.stop, dim), dim), 0, dim)
        else:
            start = np.clip(wrap_neg_index(to_int(key_val.start, dim-1), dim), -1, dim-1)
            stop = np.clip(wrap_neg_index(to_int(key_val.stop, -dim-1), dim, True), -1, dim-1)
        return slice(start, stop, step)
    else:
        # Convert to int.
        orig_key_val = to_int(key_val)
        key_val = wrap_neg_index(orig_key_val, dim)
        if 0 <= key_val < dim:
            return slice(key_val, key_val + 1, 1)
        else:
            raise IndexError(
                "Index %i is out of bounds for axis %i with size %i." % (orig_key_val, axis, dim)
            )


def to_int(val, none_val=None):
    """Convert everything but None to an int.
    """
    if val is None:
        return none_val
    else:
        return int(val)


def wrap_neg_index(index, dim, neg_step: bool = False):
    """Converts a negative index into a positive index.

    Args:
        index: The index to convert. Can be None.
        dim: The length of the dimension being indexed.
    """
    if index is not None and index < 0 and \
       not (neg_step and index == -1):
        index += dim
    return index


def index_to_slice(idx) -> slice:
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


def is_single_index(slc) -> bool:
    """Is the slice equivalent to a single index?
    """
    if slc.step is None:
        step = 1
    else:
        step = slc.step
    return slc.start is not None and \
        slc.stop is not None and \
        slc.start + step >= slc.stop


def shape(key, orig_key, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Finds the dimensions of a sliced expression.

    Args:
        key: The key used to index/slice.
        shape: The shape (row, col) of the expression.

    Returns:
        The dimensions of the expression as (rows, cols).
    """
    orig_key = to_tuple(orig_key)
    dims = []
    for i in range(len(shape)):
        if key[i] is None:
            dims.append(1)
        else:
            size = int(np.ceil((key[i].stop - key[i].start)/key[i].step))
            if size > 1 or i >= len(orig_key) or isinstance(orig_key[i], slice):
                dims.append(max(size, 0))
    return tuple(dims)


def to_str(key):
    """Converts a key (i.e. two slices) into a string.
    """
    return tuple(slice_to_str(elem) for elem in key)


def is_special_slice(key) -> bool:
    """Does the key contain a list, ndarray, or logical ndarray?
    """
    # Slices and int-like numbers are fine.
    for elem in to_tuple(key):
        if not (isinstance(elem, (numbers.Number, slice)) or np.isscalar(elem)):
            return True

    return False
