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

""" Utility functions to handle indexing/slicing into an expression. """

# Raise an Exception if the key is not a valid index.
# Returns the key as a tuple of slices.
# key - the key used to index/slice.
# shape - the shape of the expression.
def validate_key(key, shape):
    rows,cols = shape.size
    # Change single indexes for vectors into double indices.
    if not isinstance(key, tuple):
        if rows == 1:
            key = (slice(0,1,None),key)
        elif cols == 1:
            key = (key,slice(0,1,None))
        else:
            raise Exception("Invalid index %s." % key)
    # Change numbers into slices and ensure all slices have a start and step.
    key = tuple(map(format_slice, key))
    # Check that index is in bounds.
    if not (0 <= key[0].start and key[0].start < rows and \
            0 <= key[1].start and key[1].start < cols):
       raise Exception("Invalid indices %s,%s." % to_str(key))
    return key

# Utility method to convert a number to a slice and
# ensure all slices have a start and step.
# val - the value to convert into a slice.
def format_slice(val):
    if isinstance(val, slice):
        start = val.start if val.start is not None else 0
        step = val.step if val.step is not None else 1
        return slice(start, val.stop, step)
    else:
        return slice(val, val+1, 1)

# Utility method to convert a slice to a string.
def slice_to_str(slice_val):
    if is_single_index(slice_val):
        return str(slice_val.start)
    stop = slice_val.stop if slice_val.stop is not None else ''
    if slice_val.step != 1:
        return "%s:%s:%s" % (slice_val.start, stop, slice_val.step)
    else:
        return "%s:%s" % (slice_val.start, stop)

# Returns true if the slice reduces to a single index.
def is_single_index(slice_val):
    return slice_val.stop is not None and \
    slice_val.start + slice_val.step >= slice_val.stop

# Returns the stop index in the context of the expression.
# slice_val - a slice object.
# exp_dim - the length of the expression in the relevant dimension.
def get_stop(slice_val, exp_dim):
    if slice_val.stop is None:
        return exp_dim
    else:
        return min(slice_val.stop, exp_dim)

# Find the dimensions of a sliced expression.
# key - the key used to index/slice.
# shape - the shape of the expression.
def size(key, shape):
    dims = []
    for i in range(2):
        stop = get_stop(key[i], shape.size[i])
        dims.append(1 + (stop-1-key[i].start)/key[i].step)
    return tuple(dims)

# Converts a key (i.e. two slices) to a string.
def to_str(key):
    return (slice_to_str(key[0]), slice_to_str(key[1]))