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

import cvxpy.interface as intf
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class Constant(Leaf):
    """
    A constant, either matrix or scalar.
    """

    def __init__(self, value):
        # TODO HACK.
        # A fix for c.T*x where c is a 1D array.
        self.is_1D_array = False
        # Keep sparse matrices sparse.
        if intf.is_sparse(value):
            self._value = intf.DEFAULT_SPARSE_INTF.const_to_matrix(value)
            self._sparse = True
        else:
            if isinstance(value, np.ndarray) and len(value.shape) == 1:
                self.is_1D_array = True
            self._value = intf.DEFAULT_INTF.const_to_matrix(value)
            self._sparse = False
        # Set DCP attributes.
        self._size = intf.size(self.value)
        self._is_pos, self._is_neg = intf.sign(self.value)
        super(Constant, self).__init__()

    def name(self):
        return str(self.value)

    def constants(self):
        """Returns self as a constant.
        """
        return [self]

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.value]

    @property
    def value(self):
        return self._value

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    @property
    def size(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return self._size

    def is_positive(self):
        """Is the expression positive?
        """
        return self._is_pos

    def is_negative(self):
        """Is the expression negative?
        """
        return self._is_neg

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_const(self.value, self.size, self._sparse)
        return (obj, [])

    def __repr__(self):
        """Returns a string with information about the expression.
        """
        return "Constant(%s, %s, %s)" % (self.curvature,
                                         self.sign,
                                         self.size)
