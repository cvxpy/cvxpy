"""
Copyright 2018 Akshay Agrawal

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

import cvxpy.interface as intf
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.expressions.variable import Variable


class Prod(AxisAtom):
    """Multiply the entries of an expression.

    The semantics of this atom are the same as np.prod.

    This atom is log-log affine, but it is neither convex nor concave.

    Parameters
    ----------
    expr : Expression
        The expression to multiply the entries of.
    axis : int
        The axis along which to sum.
    keepdims : bool
        Whether to drop dimensions after summing.
    """

    def __init__(self, expr, axis=None, keepdims: bool = False) -> None:
        super(Prod, self).__init__(expr, axis=axis, keepdims=keepdims)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.args[0].is_nonneg():
            return (True, False)
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
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_atom_esr(self) -> bool:
        """Is the atom ESR (epigraph smooth representable)?
        """
        return True

    def is_atom_hsr(self) -> bool:
        """Is the atom HSR (hypograph smooth representable)?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def numeric(self, values):
        """Takes the product of the entries of value.
        """
        if intf.is_sparse(values[0]):
            sp_mat = values[0]
            if self.axis is None:
                if sp_mat.nnz == sp_mat.shape[0] * sp_mat.shape[1]:
                    data = sp_mat.data
                else:
                    data = np.zeros(1, dtype=sp_mat.dtype)
                result = np.prod(data)
            elif self.axis in [0, 1]:
                # The following snippet is taken from stackoverflow.
                # https://stackoverflow.com/questions/44320865/
                # can replace private _getnnz for scipy 1.15+ with count_nonzero
                mask = sp_mat._getnnz(axis=self.axis) == sp_mat.shape[self.axis]
                result = np.zeros(sp_mat.shape[1-self.axis], dtype=sp_mat.dtype)
                data = sp_mat[:, mask] if self.axis == 0 else sp_mat[mask, :]
                result[mask] = np.prod(data.toarray(), axis=self.axis)
                if self.keepdims:
                    result = np.expand_dims(result, self.axis)
            else:
                raise UserWarning("cp.prod does not support axis > 1 for sparse matrices.")
        else:
            result = np.prod(values[0], axis=self.axis, keepdims=self.keepdims)
        return result

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        return np.prod(value) / value

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _verify_jacobian_args(self):
        return isinstance(self.args[0], Variable)

    def _input_to_output_indices(self, in_shape):
        """
        Map each flattened input index to its corresponding output index.

        For axis reduction, each input element contributes to exactly one output.
        Returns array of length prod(in_shape) with output index for each input.
        """
        n_in = int(np.prod(in_shape))
        in_indices = np.arange(n_in)
        in_multi = np.array(np.unravel_index(in_indices, in_shape, order='F')).T

        if self.keepdims:
            out_multi = in_multi.copy()
            out_multi[:, self.axis] = 0
            out_shape = np.array(in_shape)
            out_shape[self.axis] = 1
        else:
            out_multi = np.delete(in_multi, self.axis, axis=1)
            out_shape = np.delete(np.array(in_shape), self.axis)

        if len(out_shape) == 0:
            return np.zeros(n_in, dtype=int)
        return np.ravel_multi_index(out_multi.T, out_shape, order='F')

    @staticmethod
    def _prod_except_self(arr):
        """
        Compute the product of all elements except each element itself.

        For arr = [a, b, c, d], returns [b*c*d, a*c*d, a*b*d, a*b*c].

        Uses prefix and suffix products to avoid division and handle zeros.
        Fully vectorized using cumulative products.
        """
        n = len(arr)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # prefix[i] = arr[0] * arr[1] * ... * arr[i-1]
        prefix = np.empty(n)
        prefix[0] = 1.0
        prefix[1:] = np.cumprod(arr[:-1])

        # suffix[i] = arr[i+1] * arr[i+2] * ... * arr[n-1]
        suffix = np.empty(n)
        suffix[-1] = 1.0
        suffix[:-1] = np.cumprod(arr[::-1])[:-1][::-1]

        return prefix * suffix

    @staticmethod
    def _prod_except_pairs(arr):
        """
        Compute the product of all elements except each pair (i, j) for i != j.

        Returns an n x n matrix H where H[i,j] = prod of arr except indices i and j.
        Diagonal entries (i == i) are set to 0.

        Returns None if all entries would be zero (3+ zeros in arr).

        Handles zeros correctly without division.
        """
        n = len(arr)
        if n == 0:
            return None
        if n == 1:
            return None

        zero_mask = (arr == 0)
        num_zeros = np.sum(zero_mask)

        if num_zeros >= 3:
            # Three or more zeros: all products are zero
            return None

        if num_zeros == 0:
            # No zeros: use prod(arr) / (arr[i] * arr[j])
            total_prod = np.prod(arr)
            H = total_prod / np.outer(arr, arr)
            np.fill_diagonal(H, 0.0)
        elif num_zeros == 1:
            # One zero at index k: only H[k, j] and H[j, k] are nonzero for j != k
            k = np.where(zero_mask)[0][0]
            prod_nonzero = np.prod(arr[~zero_mask])
            H = np.zeros((n, n))
            # H[k, j] = prod_nonzero / arr[j] for j != k
            nonzero_mask = ~zero_mask
            H[k, nonzero_mask] = prod_nonzero / arr[nonzero_mask]
            H[nonzero_mask, k] = H[k, nonzero_mask]
        else:  # num_zeros == 2
            # Two zeros: only H[k1, k2] and H[k2, k1] are nonzero
            zero_indices = np.where(zero_mask)[0]
            k1, k2 = zero_indices[0], zero_indices[1]
            prod_nonzero = np.prod(arr[~zero_mask])
            H = np.zeros((n, n))
            H[k1, k2] = prod_nonzero
            H[k2, k1] = prod_nonzero

        return H

    def _jacobian(self):
        """
        The jacobian of prod(x) with respect to x.

        For prod(x) = x_1 * x_2 * ... * x_n:
        ∂prod(x)/∂x_i = prod_{j != i}(x_j)

        Uses prefix/suffix products to handle zeros correctly.
        """
        x = self.args[0]
        x_val = x.value
        n_in = x.size
        col_idxs = np.arange(n_in, dtype=int)

        if self.axis is None:
            grad_vals = self._prod_except_self(x_val.flatten(order='F'))
            row_idxs = np.zeros(n_in, dtype=int)
        else:
            grad_vals = np.apply_along_axis(
                self._prod_except_self, self.axis, x_val
            ).flatten(order='F')
            row_idxs = self._input_to_output_indices(x.shape)

        return {x: (row_idxs, col_idxs, grad_vals)}

    def _verify_hess_vec_args(self):
        return isinstance(self.args[0], Variable)

    def _hess_vec(self, vec):
        """
        Compute weighted sum of Hessians for prod(x).

        vec has size equal to the output dimension of prod.
        For axis=None, output is scalar, so vec has size 1.
        For axis != None, vec has size equal to prod of non-reduced dimensions.

        The Hessian of prod for each output component has:
        H[i,j] = prod_{k != i,j}(x_k) for i != j (among inputs to that component)
        H[i,i] = 0

        Returns weighted combination: sum_k vec[k] * H_k
        """
        x = self.args[0]
        x_val = x.value
        n_in = x.size
        empty = (np.array([], dtype=int), np.array([], dtype=int), np.array([]))

        if self.axis is None:
            H = self._prod_except_pairs(x_val.flatten(order='F'))
            if H is None:
                return {(x, x): empty}
            H = vec[0] * H
            row_idxs, col_idxs = np.meshgrid(
                np.arange(n_in), np.arange(n_in), indexing='ij'
            )
            return {(x, x): (row_idxs.ravel(), col_idxs.ravel(), H.ravel())}

        # Multiple outputs: vec[k] weights the k-th output's Hessian
        in_indices = np.arange(n_in)
        out_idxs = self._input_to_output_indices(x.shape)
        axis_positions = np.unravel_index(in_indices, x.shape, order='F')[self.axis]
        n_out = len(np.unique(out_idxs))
        x_flat = x_val.flatten(order='F')

        all_rows = []
        all_cols = []
        all_vals = []

        for out_idx in range(n_out):
            mask = (out_idxs == out_idx)
            local_in_indices = in_indices[mask]

            # Sort by axis position to align with _prod_except_pairs
            sort_order = np.argsort(axis_positions[mask])
            sorted_in_indices = local_in_indices[sort_order]

            H_local = self._prod_except_pairs(x_flat[sorted_in_indices])
            if H_local is None:
                continue

            H_local = vec[out_idx] * H_local

            # Build global index pairs using meshgrid
            m = len(sorted_in_indices)
            local_rows, local_cols = np.meshgrid(
                np.arange(m), np.arange(m), indexing='ij'
            )
            global_rows = sorted_in_indices[local_rows.ravel()]
            global_cols = sorted_in_indices[local_cols.ravel()]
            vals = H_local.ravel()

            # Filter out zeros
            nonzero = vals != 0
            all_rows.append(global_rows[nonzero])
            all_cols.append(global_cols[nonzero])
            all_vals.append(vals[nonzero])

        if all_rows:
            return {(x, x): (
                np.concatenate(all_rows),
                np.concatenate(all_cols),
                np.concatenate(all_vals)
            )}
        return {(x, x): empty}


def prod(expr, axis=None, keepdims: bool = False) -> Prod:
    """Multiply the entries of an expression.

    The semantics of this atom are the same as np.prod.

    This atom is log-log affine, but it is neither convex nor concave.

    Parameters
    ----------
    expr : Expression or list[Expression, Numeric]
        The expression to multiply the entries of, or a list of Expressions
        and numeric types.
    axis : int
        The axis along which to take the product; ignored if `expr` is a list.
    keepdims : bool
        Whether to drop dimensions after taking the product; ignored if `expr`
        is a list.
    """
    if isinstance(expr, list):
        return Prod(hstack(expr))
    else:
        return Prod(expr, axis, keepdims)
