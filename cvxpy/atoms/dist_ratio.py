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

import numpy as np

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.length import length


class dist_ratio(Atom):
    """norm(x - a)_2 / norm(x - b)_2, with norm(x - a)_2 <= norm(x - b).

    `a` and `b` must be constants.
    """
    def __init__(self, x, a, b) -> None:
        super(dist_ratio, self).__init__(x, a, b)
        if not self.args[1].is_constant():
            raise ValueError("`a` must be a constant.")
        if not self.args[2].is_constant():
            raise ValueError("`b` must be a constant.")
        self.a = self.args[1].value
        self.b = self.args[2].value

    def numeric(self, values):
        """Returns the distance ratio.
        """
        return np.linalg.norm(
            values[0] - self.a) / np.linalg.norm(values[0] - self.b)

    shape_from_args = length.shape_from_args
    sign_from_args = length.sign_from_args
    is_atom_convex = length.is_atom_convex
    is_atom_concave = length.is_atom_concave
    is_atom_quasiconvex = length.is_atom_quasiconvex
    is_atom_quasiconcave = length.is_atom_quasiconcave
    is_incr = length.is_incr
    is_decr = length.is_decr
    _grad = length._grad
