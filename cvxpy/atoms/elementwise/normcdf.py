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

import numpy as np
import scipy as sp

from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint


class normcdf(Elementwise):
    """Elementwise :math:`\\Phi(x)` (standard normal cumulative distribution function).
    """

    def __init__(self, x) -> None:
        super(normcdf, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise standard normal cumulative distribution function of x.
        """
        return sp.stats.norm.cdf(values[0])

    def sign_from_args(self) -> tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_smooth(self) -> bool:
        """Is the atom smooth?"""
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> list[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values) -> list[Constraint]:
        """Returns the gradient of the node.
        """
        rows = self.args[0].size
        cols = self.size
        grad_vals = np.exp(-0.5 * values[0] ** 2) / np.sqrt(2 * np.pi)
        return [normcdf.elemwise_grad_to_diag(grad_vals, rows, cols)]
