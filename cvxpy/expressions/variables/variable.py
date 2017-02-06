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

from ... import settings as s
from ..leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp


class Variable(Leaf):
    """ The base variable class """
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.

    def __init__(self, rows=1, cols=1, name=None):
        self._rows = rows
        self._cols = cols
        self.id = lu.get_id()
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        else:
            self._name = name
        self.primal_value = None
        super(Variable, self).__init__()

    def is_positive(self):
        """Is the expression positive?
        """
        return False

    def is_negative(self):
        """Is the expression negative?
        """
        return False

    @property
    def size(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return (self._rows, self._cols)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._rows, self._cols, self._name]

    def name(self):
        return self._name

    def save_value(self, value):
        """Save the value of the primal variable.
        """
        value = self._validate_value(value)
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    @value.setter
    def value(self, val):
        """Assign a value to the variable.
        """
        self.save_value(val)

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {self: sp.eye(self.size[0]*self.size[1]).tocsc()}

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_var(self.size, self.id)
        return (obj, [])

    def __repr__(self):
        """String to recreate the object.
        """
        return "Variable(%d, %d)" % self.size
