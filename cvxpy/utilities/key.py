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
import math

class Key(object):
    """ A slice or index into an expression. """
    # Raise an Exception if the key is not a valid index.
    # Returns the key as a tuple of slices.
    @staticmethod
    def validate_key(key, exp):
        rows,cols = exp.size
        # Change single indexes for vectors into double indices.
        if not isinstance(key, tuple):
            if rows == 1:
                key = (slice(0,1,None),key)
            elif cols == 1:
                key = (key,slice(0,1,None))
            else:
                raise Exception("Invalid index %s for '%s'." % (key, exp.name()))
        # Change numbers into slices and ensure all slices have a start and step.
        key = tuple(map(Key.format_slice, key))
        # Check that index is in bounds.
        if not (0 <= key[0].start and key[0].start < rows and \
                0 <= key[1].start and key[1].start < cols):
           raise Exception("Invalid indices %s,%s " % Key.to_str(key) +
                           "for '%s'." % exp.name())
        return key

    # Utility method to convert a number to a slice and
    # ensure all slices have a start and step.
    @staticmethod
    def format_slice(val):
        if isinstance(val, slice):
            start = val.start if val.start is not None else 0
            step = val.step if val.step is not None else 1
            return slice(start, val.stop, step)
        else:
            return slice(val, val+1, 1)

    # Utility method to convert a slice to a string.
    @staticmethod
    def slice_to_str(slice_val):
        if Key.is_single_index(slice_val):
            return str(slice_val.start)
        stop = slice_val.stop if slice_val.stop is not None else ''
        if slice_val.step != 1:
            return "%s:%s:%s" % (slice_val.start, stop, slice_val.step)
        else:
            return "%s:%s" % (slice_val.start, stop)

    # Returns true if the slice reduces to a single index.
    @staticmethod
    def is_single_index(slice_val):
        return slice_val.stop is not None and \
        slice_val.start + slice_val.step >= slice_val.stop

    # Returns the stop index in the context of the expression.
    @staticmethod
    def get_stop(slice_val, exp_dim):
        if slice_val.stop is None:
            return exp_dim
        else:
            return min(slice_val.stop, exp_dim)

    # Find the dimensions of a sliced expression.
    @staticmethod
    def size(key, exp):
        dims = []
        for i in range(2):
            stop = Key.get_stop(key[i], exp.size[i])
            dims.append(1 + (stop-1-key[i].start)/key[i].step)
        return tuple(dims)

    # Converts a key (i.e. two slices) to a string.
    @staticmethod
    def to_str(key):
        return (Key.slice_to_str(key[0]), Key.slice_to_str(key[1]))

    # Returns a key that's the composition of two keys.
    @staticmethod
    def compose_keys(new_key, old_key):
        return (Key.compose_slices(new_key[0], old_key[0]),
                Key.compose_slices(new_key[1], old_key[1]))

    # Returns a slice that's the composition of two slices.
    @staticmethod
    def compose_slices(new_slc, old_slc):
        start = old_slc.start + new_slc.start*old_slc.step
        step_size = step_size = new_slc.step*old_slc.step
        if new_slc.stop is None and old_slc.stop is None:
            stop = None
        elif new_slc.stop is None:
            steps = float(old_slc.stop - old_slc.step*new_slc.start)/step_size
            # Round up so that the last step is taken.
            stop = int(math.ceil(steps*step_size + start))
        else:
            stop = old_slc.start + new_slc.stop*old_slc.step
        return slice(start, stop, step_size)