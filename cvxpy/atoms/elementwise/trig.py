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

    def sign_from_args(self) -> tuple[bool, bool]:
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
        return False

    def is_atom_smooth(self) -> bool:
        """Is the atom smooth?"""
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

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

    def sign_from_args(self) -> tuple[bool, bool]:
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
        return False

    def is_atom_smooth(self) -> bool:
        """Is the atom smooth?"""
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

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

    def sign_from_args(self) -> tuple[bool, bool]:
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
        return False

    def is_atom_smooth(self) -> bool:
        """Is the atom smooth?"""
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> list[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [-np.pi/2 <= self.args[0], self.args[0] <= np.pi/2]

    def _grad(self, values) -> list[Constraint]:
        """Returns the gradient of the node.
        """
        rows = self.args[0].size
        cols = self.size
        grad_vals = 1/np.cos(values[0])**2
        return [tan.elemwise_grad_to_diag(grad_vals, rows, cols)]


class atan(Elementwise):
    """Elementwise :math:`\\arctan x`.
    """

    def __init__(self, x) -> None:
        super(atan, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise arctangent of x.
        """
        return np.arctan(values[0])

    def sign_from_args(self) -> tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # atan is odd and increasing, so it has the sign of its argument.
        return (self.args[0].is_nonneg(), self.args[0].is_nonpos())

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
        grad_vals = 1/(1 + values[0]**2)
        return [atan.elemwise_grad_to_diag(grad_vals, rows, cols)]


class atan2(Elementwise):
    """Elementwise :math:`\\operatorname{atan2}(y, x)`, the angle of the point
    :math:`(x, y)` in :math:`(-\\pi, \\pi]`.

    The argument order follows C and NumPy: ``atan2(y, x)``.
    """

    def __init__(self, y, x) -> None:
        super(atan2, self).__init__(y, x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the elementwise angle of the point (x, y).
        """
        return np.arctan2(values[0], values[1])

    def sign_from_args(self) -> tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # y >= 0 gives an angle in [0, pi]. y <= 0 alone is not enough for a
        # nonpositive angle since atan2(0, -1) = pi; with x >= 0 the angle
        # lies in [-pi/2, 0].
        return (self.args[0].is_nonneg(),
                self.args[0].is_nonpos() and self.args[1].is_nonneg())

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
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _domain(self) -> list[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        # The domain is R^2 minus the origin, which has no constraint form.
        return []

    def _grad(self, values) -> list[Constraint]:
        """Returns the gradient of the node.
        """
        y = values[0]
        x = values[1]
        r2 = y**2 + x**2
        if np.any(r2 == 0):
            # Non-differentiable at the origin.
            return [None, None]
        cols = self.size
        return [atan2.elemwise_grad_to_diag(x / r2, self.args[0].size, cols),
                atan2.elemwise_grad_to_diag(-y / r2, self.args[1].size, cols)]
