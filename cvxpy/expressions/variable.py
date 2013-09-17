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

import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
import expression
import cvxpy.utilities as u
import leaf

class Variable(leaf.Leaf):
    """ The base variable class """
    VAR_COUNT = 0        
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    # value_matrix - the matrix type used to store values.
    def __init__(self, rows=1, cols=1, name=None, value_matrix=intf.DENSE_TARGET):
        self._context = u.Context(u.Sign.UNKNOWN, u.Curvature.AFFINE, u.Shape(rows, cols))
        self._init_id()
        self._name = self.id if name is None else name
        self.interface = intf.get_matrix_interface(value_matrix)
        self.primal_value = None
        super(Variable, self).__init__()

    # Initialize the id.
    def _init_id(self):
        self.id = Variable.next_var_name()

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        Variable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self._name

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Returns the id's of the index variables, each with a matrix
    # of the same dimensions as the variable that is 0 except
    # for at the index key, where it is 1.
    def coefficients(self, interface):
        coeffs = {}
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                id = self.index_id(row, col)
                # For scalars, coefficient must be a number.
                if self.size == (1,1):
                    coeff = 1
                else:
                    coeff = interface.zeros(*self.size)
                    coeff[row,col] = 1
                coeffs[id] = coeff
        return coeffs

    # Return self.
    def variables(self):
        return [self]

    # The id of the view at the given index.
    def index_id(self, row, col):
        return "%s[%s,%s]" % (self.id, row, col)

    # Return a scalar view into a matrix variable.
    def index_object(self, key):
        return IndexVariable(self, key)

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