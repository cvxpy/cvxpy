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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants.parameter import is_param_free

if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint


class quad_over_root(Atom):
    r"""The function :math:`(ax^2 + bx + c) / \sqrt{y}`.

    This atom represents the scalar-valued function

    .. math::
        f(x, y) = \frac{ax^2 + bx + c}{\sqrt{y}}

    on the domain :math:`x \geq d`, :math:`y > 0`.

    The function is jointly convex in :math:`(x, y)` when :math:`a > 0` and
    the convexity condition

    .. math::
        2a^2 d^2 + 2abd + 6ac - b^2 \geq 0

    holds. When ``d`` is ``None``, the atom requires the numerator to be
    globally nonneg (:math:`4ac \geq b^2`), and the domain constraint on
    ``x`` is omitted.

    Parameters
    ----------
    x : Expression
        A scalar expression.
    y : Expression
        A scalar expression (must be positive).
    a : float
        Positive coefficient of :math:`x^2`.
    b : float
        Coefficient of :math:`x`.
    c : float
        Constant term.
    d : float or None
        Lower bound on ``x``. When ``None``, the numerator must be globally
        nonneg.

    This atom is useful for a continuous relaxation of a continuous quadratic * a binary variable. This formulation, rather than simply relaxing the binary variable to be in [0, 1], can be convex when the convexity condition is satisfied.
    Small y corresponds to the binary variable being 1, and large y corresponds to the binary variable being 0. The convexity condition ensures that the function is convex in this regime.
    In general 
    .. math::
        f(x, y) = \frac{ax^2 + bx + c}{y^\alpha}
    is convex for :math:`\alpha \in (0, 1]` when certain conditions are met. :math:`\alpha = 1/2` is the least restrictive choice for convexity, and is the choice implemented in this atom.
    .. math::
        det(\del^2 f(x,y)) = \frac{2a^2x^2 (\alpha-\alpha^2) + 2abx (\alpha-\alpha^2) + [2ac(\alpha+\alpha^2) - b^2 \alpha^2]}{y^{2\alpha+2}}
    The first two terms are maximized over :math:`\alpha` at :math:`\alpha=0.5`. The maximum of the last term depends on a, b, and c. We therefore choose :math:`\alpha=0.5` to make the convexity condition as weak as possible.
    For example, for a practical unit commitment problem, the convexity condition is almost never satisified with :math:`\alpha=1`, but almost always satisfied with :math:`\alpha=0.5`.
    """

    def __init__(self, x, y, a: float, b: float, c: float,
                 d: float | None = None) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = None if d is None else float(d)
        super().__init__(x, y)

    def get_data(self):
        return [self.a, self.b, self.c, self.d]

    def copy(self, args=None, id_objects=None):
        if args is None:
            args = self.args
        return type(self)(args[0], args[1], self.a, self.b, self.c, self.d)

    def validate_arguments(self) -> None:
        if not self.args[0].is_scalar():
            raise ValueError("The first argument to quad_over_root must be a scalar.")
        if not self.args[1].is_scalar():
            raise ValueError("The second argument to quad_over_root must be a scalar.")
        if self.args[0].is_complex() or self.args[1].is_complex():
            raise ValueError("Arguments to quad_over_root cannot be complex.")
        if self.a <= 0:
            raise ValueError("Coefficient a must be positive.")

        disc = self.b ** 2 - 4 * self.a * self.c
        if self.d is None:
            if disc > 1e-10:
                raise ValueError(
                    "When d is None, require 4ac >= b^2 "
                    "(globally nonneg numerator)."
                )
        else:
            # Check numerator is nonneg at x=d.
            p_d = self.a * self.d ** 2 + self.b * self.d + self.c
            if p_d < -1e-10:
                raise ValueError(
                    f"Numerator is negative at x=d={self.d}: p(d)={p_d}."
                )
            # Check convexity condition.
            g_d = (2 * self.a ** 2 * self.d ** 2
                   + 2 * self.a * self.b * self.d
                   + 6 * self.a * self.c
                   - self.b ** 2)
            if g_d < -1e-10:
                raise ValueError(
                    "Convexity condition 2a^2 d^2 + 2abd + 6ac - b^2 >= 0 "
                    f"not satisfied: g(d) = {g_d}."
                )

    @Atom.numpy_numeric
    def numeric(self, values):
        x_val = np.asarray(values[0]).item()
        y_val = np.asarray(values[1]).item()
        return np.array(
            (self.a * x_val ** 2 + self.b * x_val + self.c) / np.sqrt(y_val)
        )

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        # Nonneg on the domain.
        return (True, False)

    def is_atom_convex(self) -> bool:
        if u.scopes.dpp_scope_active():
            return is_param_free(self.args[0]) and is_param_free(self.args[1])
        return True

    def is_atom_concave(self) -> bool:
        return False

    def is_atom_smooth(self) -> bool:
        return True

    def is_incr(self, idx: int) -> bool:
        return False

    def is_decr(self, idx: int) -> bool:
        # Decreasing in y (for nonneg numerator).
        return idx == 1

    def _domain(self) -> list[Constraint]:
        constraints: list[Constraint] = [self.args[1] >= 0]
        if self.d is not None:
            constraints.append(self.args[0] >= self.d)
        return constraints

    def _grad(self, values):
        return [None, None]
