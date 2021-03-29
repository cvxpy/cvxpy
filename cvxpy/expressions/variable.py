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

from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp


def upper_tri_to_full(n):
    """Returns a coefficient matrix to create a symmetric matrix.

    Parameters
    ----------
    n : int
        The width/height of the matrix.

    Returns
    -------
    SciPy CSC matrix
        The coefficient matrix.
    """
    entries = n*(n+1)//2

    val_arr = []
    row_arr = []
    col_arr = []
    count = 0
    for i in range(n):
        for j in range(i, n):
            # Index in the original matrix.
            col_arr.append(count)
            # Index in the filled matrix.
            row_arr.append(j*n + i)
            val_arr.append(1.0)
            if i != j:
                # Index in the original matrix.
                col_arr.append(count)
                # Index in the filled matrix.
                row_arr.append(i*n + j)
                val_arr.append(1.0)
            count += 1

    return sp.csc_matrix((val_arr, (row_arr, col_arr)),
                         (n*n, entries))


class Variable(Leaf):
    """The optimization variables in a problem.
    """

    def __init__(self, shape=(), name=None, var_id=None, **kwargs):
        if var_id is None:
            self.id = lu.get_id()
        else:
            self.id = var_id
        if name is None:
            self._name = "%s%d" % (s.VAR_PREFIX, self.id)
        elif isinstance(name, str):
            self._name = name
        else:
            raise TypeError("Variable name %s must be a string." % name)

        self._variable_with_attributes = None
        self._value = None
        self.delta = None
        self.gradient = None
        super(Variable, self).__init__(shape, **kwargs)

    def name(self):
        """str : The name of the variable."""
        return self._name

    def is_constant(self) -> bool:
        return False

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # TODO(akshayka): Do not assume shape is 2D.
        return {self: sp.eye(self.size).tocsc()}

    def variables(self):
        """Returns itself as a variable.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_var(self.shape, self.id)
        return (obj, [])

    def attributes_were_lowered(self):
        """True iff variable generated when lowering a variable with attributes.
        """
        return self._variable_with_attributes is not None

    def set_variable_of_provenance(self, variable):
        assert variable.attributes
        self._variable_with_attributes = variable

    def variable_of_provenance(self):
        """Returns a variable with attributes from which this variable was generated."""
        return self._variable_with_attributes

    def __repr__(self):
        """String to recreate the object.
        """
        attr_str = self._get_attr_str()
        if len(attr_str) > 0:
            return "Variable(%s%s)" % (self.shape, attr_str)
        else:
            return "Variable(%s)" % (self.shape,)
