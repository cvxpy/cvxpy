"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants.parameter import is_param_free
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import scopes


class perspective(Atom):
    r"""Implements the perspective transform of a convex or concave scalar
    expression. Uses the fact that given a cone form for the epigraph of :math:`f` via

    :math:`\{ (t, x) \in \mathbb{R}^{n+1} : t \geq f(x) \}`
    :math:`= \{ (t,x) : Fx + gt + e \in K \},`

    the epigraph of the perspective transform
    of f can be given by

    :math:`\{ (t, x, s) \in \mathbb{R}^{n+2} : t \geq sf(x/s) \}`
    :math:`= \{ (t,x,s) : Fx + gt + se \in K \},`

    (see https://web.stanford.edu/~boyd/papers/pdf/sw_aff_ctrl.pdf).

    Note that this is the perspective transform of a scalar expression viewed as
    a function of its underlying variables. The perspective atom does not return
    a `Callable`, so you cannot create compositions such as :math:`p(g(x),s)`, where
    :math:`p(z,s) = sf(z/s)` is the perpective transform of :math:`f`.
    """

    def __init__(self, f: Expression, s: Variable) -> None:
        self.f = f
        super(perspective, self).__init__(s, *f.variables())

    def validate_arguments(self) -> None:
        assert self.f.size == 1  # dealing only with scalars, for now
        assert self.args[0].size == 1
        assert isinstance(self.args[0], Variable), "s must be a variable"
        assert self.args[0].is_nonneg(), "s must be a nonnegative variable"
        return super().validate_arguments()

    def numeric(self, values: list[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the perspective sf(x/s) numerically.
        """

        assert values[0] >= 0
        assert not np.isclose(values[0], 0.0), \
            "There are valid cases where s = 0, but we do not handle this yet, e.g., f(x) = x + 1."

        s_val = np.array(values[0])
        f = self.f

        old_x_vals = [var.value for var in f.variables()]

        def set_vals(vals, s_val):
            for var, val in zip(f.variables(), vals):
                var.value = val/s_val

        set_vals(values[1:], s_val=values[0])

        ret_val = np.array([f.value*s_val])

        set_vals(old_x_vals, s_val=1)

        return ret_val

    def sign_from_args(self) -> tuple[bool, bool]:
        f_pos = self.f.is_nonneg()
        f_neg = self.f.is_nonpos()
        s_pos = self.args[0].is_nonneg()

        assert s_pos

        is_positive = (f_pos and s_pos)
        is_negative = (f_neg and s_pos)

        return is_positive, is_negative

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        if scopes.dpp_scope_active() and not is_param_free(self.f):
            return False
        else:
            return self.f.is_convex() and self.args[0].is_nonneg()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        if scopes.dpp_scope_active() and not is_param_free(self.f):
            return False
        else:
            return self.f.is_concave() and self.args[0].is_nonneg()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx: int) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.f.shape
