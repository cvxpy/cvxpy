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
    # Change single indexes for vectors into double indices.
    if not isinstance(key, tuple):
        if rows == 1:
            key = (slice(0, 1, None), key)
        elif cols == 1:
            key = (key, slice(0, 1, None))
        else:
            raise IndexError("Invalid index/slice.")
    # Change numbers into slices and ensure all slices have a start and step.
    key = tuple(format_slice(slc, dim) for slc, dim in zip(key, shape.size))
    # Check that index is in bounds.
    if not (0 <= key[0].start and key[0].start < rows and \
            0 <= key[1].start and key[1].start < cols):
        raise IndexError("Index/slice out of bounds.")
    return key

def format_slice(key_val, dim):
    """Converts part of a key into a slice with a start and step.

    Args:
        key_val: The value to convert into a slice.
        dim: The length of the dimension being sliced.

    Returns:
        A slice with a start and step.
    """
    if isinstance(key_val, slice):
        start = key_val.start if key_val.start is not None else 0
        step = key_val.step if key_val.step is not None else 1
        return slice(wrap_neg_index(start, dim),
                     wrap_neg_index(key_val.stop, dim),
                     step)
    else:
        key_val = wrap_neg_index(key_val, dim)
        return slice(key_val, key_val+1, 1)

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

def slice_to_str(slice_):
    """Converts a slice into a string.
    """
    if is_single_index(slice_):
        return str(slice_.start)
    stop = slice_.stop if slice_.stop is not None else ''
    if slice_.step != 1:
        return "%s:%s:%s" % (slice_.start, stop, slice_.step)
    else:
        return "%s:%s" % (slice_.start, stop)

def is_single_index(slice_):
    """Is the slice equivalent to a single index?
    """
    return slice_.stop is not None and \
    slice_.start + slice_.step >= slice_.stop

def get_stop(slice_, exp_dim):
    """Returns the stopping index for the slice applied to the expression.

    Args:
        slice_: A Slice into the expression.
        exp_dim: The length of the expression along the sliced dimension.

    Returns:
        The stopping index for the slice applied to the expression.
    """
    if slice_.stop is None:
        return exp_dim
    else:
        return min(slice_.stop, exp_dim)

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
        stop = get_stop(key[i], shape.size[i])
        dims.append(1 + (stop-1-key[i].start)/key[i].step)
    return tuple(dims)

def to_str(key):
    """Converts a key (i.e. two slices) into a string.
    """
    return (slice_to_str(key[0]), slice_to_str(key[1]))
