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


class sin(Elementwise):
    """Elementwise :math:`\\sin x`.
    """

    def __init__(self, x) -> None:
        super(sin, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise sine of x.
        """
        return np.sin(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values) -> List[Constraint]:
        """Returns the gradient of the node.
        """
        rows = self.args[0].size
        cols = self.size
        grad_vals = np.cos(values[0])
        return [sin.elemwise_grad_to_diag(grad_vals, rows, cols)]


class cos(Elementwise):
    """Elementwise :math:`\\cos x`.
    """

    def __init__(self, x) -> None:
        super(cos, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise cosine of x.
        """
        return np.cos(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values) -> List[Constraint]:
        """Returns the gradient of the node.
        """
        rows = self.args[0].size
        cols = self.size
        grad_vals = -np.sin(values[0])
        return [cos.elemwise_grad_to_diag(grad_vals, rows, cols)]


class tan(Elementwise):
    """Elementwise :math:`\\tan x`.
    """

    def __init__(self, x) -> None:
        super(tan, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise tangent of x.
        """
        return np.tan(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always unknown.
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return []

    def _grad(self, values) -> List[Constraint]:
        """Returns the gradient of the node.
        """
        rows = self.args[0].size
        cols = self.size
        grad_vals = 1/np.cos(values[0])**2
        return [tan.elemwise_grad_to_diag(grad_vals, rows, cols)]
