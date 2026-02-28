"""
Copyright 2013 Steven Diamond

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

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.real import real as real_atom
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint


def trace(expr):
    """
    TLDR: Use alternate formulation for trace(A@B) for more efficient computation.
    trace(A@B) normally is O(n^3) because of the A@B operation.
    However, trace(A@B) only requires diagonal entries of A@B, which can be
    computed in O(n^2) time using the identity:

        trace(A @ B) = sum_ij A_ij * B_ji = sum(A * B.T)

    This avoids forming the full matrix product while remaining correct for
    both real and complex matrices (no conjugation on A is needed).

    When the MulExpression is Hermitian (e.g. X @ X^H for Hermitian X), the
    result is provably real; it is wrapped with real() so that is_real()
    correctly returns True without routing to the O(n^3) Trace(expr) path.
    """
    if isinstance(expr, MulExpression):
        # trace(A @ B) = sum(A * B.T), correct for real and complex matrices.
        result = cvxpy_sum(multiply(expr.args[0], expr.args[1].T))
        if expr.is_hermitian():
            # trace of a Hermitian matrix is provably real.
            # Wrap with real() so is_real() == True propagates upward.
            return real_atom(result)
        return result
    else:
        return Trace(expr)


class Trace(AffAtom):
    """The sum of the diagonal entries of a matrix.

    Follows ``np.linalg.trace`` conventions: for an input with shape
    ``(*batch, n, n)``, returns an expression with shape ``(*batch,)``.

    Parameters
    ----------
    expr : Expression
        The expression to sum the diagonal of.
    """

    def __init__(self, expr) -> None:
        super(Trace, self).__init__(expr)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Trace is nonneg (nonpos) if its argument is elementwise nonneg
        (nonpos) or psd (nsd).
        """
        is_nonneg = self.args[0].is_nonneg() or self.args[0].is_psd()
        is_nonpos = self.args[0].is_nonpos() or self.args[0].is_nsd()

        return is_nonneg, is_nonpos

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Sums the diagonal entries.
        """
        return np.linalg.trace(values[0])

    def validate_arguments(self) -> None:
        """Checks that the argument is a square matrix (possibly batched).
        """
        shape = self.args[0].shape
        if self.args[0].ndim < 2 or shape[-2] != shape[-1]:
            raise ValueError(
                "Argument to trace must be a square array with ndim >= 2."
            )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Scalar for 2D input, batch shape for ND input.
        """
        return self.args[0].shape[:-2]

    def is_real(self) -> bool:
        return self.args[0].is_real() or self.args[0].is_hermitian()

    def is_complex(self) -> bool:
        return not self.is_real()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Sum the diagonal entries of the linear expression.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.trace(arg_objs[0], shape), [])
