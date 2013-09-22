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

from variable import Variable

class IndexVariable(Variable):
    """ An index into a matrix variable """
    # parent - the variable indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        name = "%s[%s,%s]" % (parent.name(), key[0], key[1])
        super(IndexVariable, self).__init__(name=name)

    # Coefficient is always 1.
    def coefficients(self, interface):
        return {self.id: 1}

    # Return parent so that the parent value is updated.
    def variables(self):
        return [self.parent]

    # Initialize the id.
    def _init_id(self):
        self.id = self.parent.index_id(*self.key)

    # Convey the parent's constraints to the canonicalization.
    def _constraints(self):
        return self.parent._constraints()

    # The value at the index.
    @property
    def value(self):
        if self.parent.value is None:
            return None
        else:
            return self.parent.value[self.key]