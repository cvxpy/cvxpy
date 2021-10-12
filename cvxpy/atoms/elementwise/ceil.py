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
from typing import Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.atoms.elementwise.elementwise import Elementwise


class ceil(Elementwise):
    """Elementwise ceiling."""

    def __init__(self, x) -> None:
        return super(ceil, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        decimals = int(np.abs(np.log10(s.ATOM_EVAL_TOL)))
        return np.ceil(np.around(values[0], decimals=decimals))

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.args[0].is_nonneg() and self.args[0].is_nonpos():
            return (True, True)
        elif self.args[0].is_nonneg():
            return (True, False)
        elif self.args[0].is_nonpos():
            return (False, True)
        else:
            return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return True

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconcave?
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

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return sp.csc_matrix(self.args[0].shape)


class floor(Elementwise):
    """Elementwise floor."""

    def __init__(self, x) -> None:
        return super(floor, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.floor(values[0])

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.args[0].is_nonneg() and self.args[0].is_nonpos():
            return (True, True)
        elif self.args[0].is_nonneg():
            return (True, False)
        elif self.args[0].is_nonpos():
            return (False, True)
        else:
            return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return True

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconcave?
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

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return sp.csc_matrix(self.args[0].shape)
