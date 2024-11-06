"""
Copyright 2023, the CVXPY authors

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
from scipy import linalg as LA
from scipy.stats import entropy

from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint


class quantum_rel_entr(Atom):
    """
    Parameters
    ----------
    X : Expression or numeric
        A PSD matrix
    Y : Expression or numeric
        A PSD matrix
    """

    def __init__(self, X, Y, quad_approx: Tuple[int, int] = (3, 3)) -> None:
        # TODO: add a check that N is symmetric/Hermitian.
        self.quad_approx = quad_approx
        super(quantum_rel_entr, self).__init__(X, Y)

    def numeric(self, values):
        X, Y = values
        if hasattr(X, 'value') and hasattr(Y, 'value'):
            X = X.value
            Y = Y.value
        # Symmetrize A and B
        X = (X + X.conj().T) / 2
        Y = (Y + Y.conj().T) / 2
        w1, V = LA.eigh(X)
        w2, W = LA.eigh(Y)
        u = w1.T @ np.abs(V.conj().T @ W) ** 2
        def func(x):
            assert np.all(x >= 0)
            x_sum = x.sum()
            val = -entropy(x)
            un_normalized = (x_sum * val + np.log(x_sum)*x_sum)
            return un_normalized
        r1 = -entropy(w1)
        r2 = u @ np.log(w2)
        return (r1 - r2)

    def validate_arguments(self) -> None:
        if not (self.args[0].is_hermitian() and self.args[1].is_hermitian()):
            raise ValueError('Arguments must be Hermitian')

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        return tuple()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def get_data(self):
        return [self.quad_approx]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        raise NotImplementedError()

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >> 0, self.args[1] >> 0]
