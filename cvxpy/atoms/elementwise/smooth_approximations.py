"""
Copyright 2025 CVXPY Developers

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
from typing import List, Tuple

import numpy as np

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable


class smooth_abs(Elementwise):
    """Elementwise :math:`(1/a) * log (exp(ax) + exp(-ax))`.
    """

    def __init__(self, x) -> None:
        super(smooth_abs, self).__init__(x)
        self.a = 5.0

    def numeric(self, values):
        x = values[0]
        results = np.logaddexp(self.a*x, -self.a*x) / self.a
        # Return -inf outside the domain
        if np.isscalar(results):
            if np.isnan(results):
                return -np.inf
        else:
            results[np.isnan(results)] = -np.inf
        return results

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        raise NotImplementedError()
        

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        raise NotImplementedError()
        

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
        grad_vals = np.tanh(self.a * values[0])
        return [smooth_abs.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        """ See the docstring of the hess_vec method of the atom class. """
        x = self.args[0]
        t = np.tanh(self.a * x.value)
        hess_vals = self.a * (1 - t**2)
        return {(x, x): np.diag(vec * hess_vals)}

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        raise NotImplementedError()
    
    def point_in_domain(self):
        raise NotImplementedError()
