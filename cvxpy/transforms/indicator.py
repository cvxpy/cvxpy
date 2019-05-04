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

from cvxpy.expressions.expression import Expression
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class indicator(Expression):
    """An expression representing the convex function I(constraints) = 0
       if constraints hold, +infty otherwise.

    Parameters
    ----------
    constraints : list
       A list of constraint objects.
    err_tol:
       A numeric tolerance for determining whether the constraints hold.
    """

    def __init__(self, constraints, err_tol=1e-3):
        self.args = constraints
        self.err_tol = err_tol
        super(indicator, self).__init__()

    def is_convex(self):
        """Is the expression convex?
        """
        return True

    def is_concave(self):
        """Is the expression concave?
        """
        return False

    def is_log_log_convex(self):
        return False

    def is_log_log_concave(self):
        return False

    def is_nonneg(self):
        """Is the expression positive?
        """
        return True

    def is_nonpos(self):
        """Is the expression negative?
        """
        return False

    def is_imag(self):
        """Is the Leaf imaginary?
        """
        return False

    def is_complex(self):
        """Is the Leaf complex valued?
        """
        return False

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.err_tol]

    @property
    def shape(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return ()

    def name(self):
        """Returns the string representation of the expression.
        """
        return "Indicator(%s)" % str(self.args)

    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        return self.args

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        if all(cons.value() for cons in self.args):
            return 0
        else:
            return np.infty

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.
        None indicates variable values unknown or outside domain.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        # TODO
        return NotImplemented

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        constraints = []
        for cons in self.args:
            constraints += cons.canonical_form[1]
        return (lu.create_const(0, (1, 1)), constraints)
