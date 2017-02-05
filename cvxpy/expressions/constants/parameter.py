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

from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu


class Parameter(Leaf):
    """
    A parameter, either matrix or scalar.
    """
    PARAM_COUNT = 0

    def __init__(self, rows=1, cols=1, name=None, sign="unknown", value=None):
        self.id = lu.get_id()
        self._rows = rows
        self._cols = cols
        self._sign_str = sign
        if name is None:
            self._name = "%s%d" % (s.PARAM_PREFIX, self.id)
        else:
            self._name = name
        # Initialize with value if provided.
        self._value = None
        if value is not None:
            self.value = value
        super(Parameter, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._rows, self._cols, self._name, self._sign_str, self.value]

    def name(self):
        return self._name

    @property
    def size(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return (self._rows, self._cols)

    def is_positive(self):
        """Is the expression positive?
        """
        return self._sign_str == s.ZERO or self._sign_str.upper() == s.POSITIVE

    def is_negative(self):
        """Is the expression negative?
        """
        return self._sign_str == s.ZERO or self._sign_str.upper() == s.NEGATIVE

    # Getter and setter for parameter value.
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = self._validate_value(val)

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    def parameters(self):
        """Returns itself as a parameter.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_param(self, self.size)
        return (obj, [])

    def __repr__(self):
        """String to recreate the object.
        """
        return 'Parameter(%d, %d, sign="%s")' % (self._rows,
                                                 self._cols,
                                                 self.sign)
