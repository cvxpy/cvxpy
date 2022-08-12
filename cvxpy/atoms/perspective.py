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

from cvxpy.atoms.atom import Atom
import cvxpy as cp
from typing import Tuple
import numpy as np


class perspective(Atom):
    """TODO.
    """

    def __init__(self, f: cp.Expression, s: cp.Expression) -> None:
        self.f = f
        super(perspective, self).__init__(s, *f.variables())

    def validate_arguments(self) -> None:
        assert self.f.size == 1  # dealing only with scalars, for now
        # assert self.args[0].size == 1
        assert self.args[0].size == 1

        # assert self.f.variables() == [self.args[0]]

        return super().validate_arguments()

    def numeric(self, values):
        """
        Compute the perspective sf(x/s) numerically.
        """

        assert values[0] >= 0

        s_val = np.array(values[0])
        f = self.f

        # TODO: fix this silly overwriting
        # old_x_val = self.args[0].value
        old_x_vals = [var.value for var in f.variables()]

        def new_set_vals(vals, s_val):
            for var, val in zip(f.variables(), vals):
                var.value = val/s_val 

        def set_vals(vals, s_val=1):
            # vals could be scalar, could be an array

            vals = np.atleast_1d(vals)
            i = 0
            for var in f.variables():
                d = int(np.prod(var.shape))
                new_val = (vals[i:i+d]/s_val).reshape(var.shape)
                var.value = new_val
                i += d

            # n = len(f.variables())
            # vals = vals.reshape((n, -1))
            # for var, val in zip(f.variables(), vals):
            #     new_val = np.array(val/s_val).reshape(var.shape)
            #     var.value = new_val

        new_set_vals(values[1:], s_val=values[0])

        ret_val = np.array([f.value*s_val])

        new_set_vals(old_x_vals, s_val=1)

        return ret_val

    def _grad(self, values):
        """
        """
        pass

    def _column_grad(self, value):
        pass

    def sign_from_args(self) -> Tuple[bool, bool]:
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
        return self.f.is_convex() and self.args[0].is_nonneg()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return self.f.is_concave() and self.args[0].is_nonneg()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        assert idx in [1, 2], "can't handle increasing in 'f'"
        if idx == 1:
            return self.f.is_incr(0)  # assuming scalar for now
        elif idx == 2:
            return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        assert idx in [0, 1]
        if idx == 0:
            return self.f.is_decr(0)  # assuming scalar for now
        elif idx == 1:
            return True
        pass

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.f.shape
