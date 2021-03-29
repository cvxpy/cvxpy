"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.atoms.elementwise.log import log
import scipy
import numpy as np


class log1p(log):
    """Elementwise :math:`\\log (1 + x)`.
    """

    def __init__(self, x) -> None:
        super(log1p, self).__init__(x)

    @log.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise natural log of x+1.
        """
        return scipy.special.log1p(values[0])

    def sign_from_args(self):
        """The same sign as the argument.
        """
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size
        # Outside domain or on boundary.
        if np.min(values[0]) <= -1:
            # Non-differentiable.
            return [None]
        else:
            grad_vals = 1.0/(values[0]+1)
            return [log1p.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= -1]
