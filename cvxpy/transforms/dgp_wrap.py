"""
Copyright, the CVXPY authors

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
from cvxpy.utilities import performance_utils as perf
import numpy as np


class dgp_wrap(Expression):
    """Converts the function f(x) into exp(f(log x)).
       Under the DGP to DCP transformation, you get back the original
       function. Hence, all the convexity properties of the function
       become log-log convexity properties.


    Parameters
    ----------
    function :
       An atomic function
    args : list
       The arguments to the wrapped function.
    """

    def __init__(self, function, args):
        self.function = function
        self.args = args
        self.expr = self.function(*args)
        super(dgp_wrap, self).__init__()

    def is_convex(self):
        """Is the expression convex?
        """
        return False

    def is_concave(self):
        """Is the expression concave?
        """
        return False

    @perf.compute_once
    def is_log_log_convex(self):
        """Is the expression log-log convex?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_convex():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_convex() and self.is_incr(idx)) or
                        (arg.is_concave() and self.is_decr(idx))):
                    return False
            return True
        else:
            return False

    @perf.compute_once
    def is_log_log_concave(self):
        """Is the expression log-log concave?
        """
        # Applies DCP composition rule.
        if self.is_constant():
            return True
        elif self.is_atom_concave():
            for idx, arg in enumerate(self.args):
                if not (arg.is_affine() or
                        (arg.is_concave() and self.is_incr(idx)) or
                        (arg.is_convex() and self.is_decr(idx))):
                    return False
            return True
        else:
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
        return [self.function]

    @property
    def shape(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return self.expr.shape

    def name(self):
        """Returns the string representation of the expression.
        """
        return "dgp_wrap(%s, %s)" % (str(self.function), str(self.args))

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
        values = [arg.value for arg in self.args]
        if any([v is None for v in values]):
            return None
        else:
            log_vals = [np.log(v) for v in values]
            return np.exp(self.expr.numeric(log_vals))

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
