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

class Key(object):
    """ A slice or index into an expression. """
    # Raise an Exception if the key is not a valid index.
    # Returns the key as a tuple.
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
        # Change numbers into slices.
        key = (Key.num_to_slice(key[0]), Key.num_to_slice(key[1]))
        # Check that index is in bounds.
        if not (0 <= key[0].start and key[0].start < rows and \
                0 <= key[1].start and key[1].start < cols):
           raise Exception("Invalid indices %s,%s " % Key.to_str(key) +
                           "for '%s'." % exp.name())
        return key

    # Utility method to convert a number to a slice.
    @staticmethod
    def num_to_slice(num):
        if isinstance(num, slice):
            return num
        else:
            return slice(num, num+1, None)

    # Utility method to convert a slice to a string.
    @staticmethod
    def slice_to_str(slice_val):
        values = [slice_val.start, slice_val.stop, slice_val.step]
        step_size = values[2] if values[2] else 1
        if values[0] + step_size >= values[1]:
            return str(values[0])
        values = [v if v else '' for v in values]
        if slice_val.step:
            return "%s:%s:%s" % tuple(values)
        elif slice_val.stop:
            return "%s:%s" % tuple(values[0:1])
        else:
            return str(values[0])

    # Converts a key (i.e. two slices) to a string.
    @staticmethod
    def to_str(key):
        return (Key.slice_to_str(key[0]), Key.slice_to_str(key[1]))