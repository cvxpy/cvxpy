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
from scipy.sparse import coo_matrix

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


def vstack(arg_list) -> "Vstack":
    """Wrapper on vstack to ensure list argument.
    """
    return Vstack(*arg_list)


class Vstack(AffAtom):
    """Vertical concatenation"""
    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    # Returns the vstack of the values.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.vstack(values)

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self) -> Tuple[int, ...]:
        try:
            return np.vstack(
                [np.empty(arg.shape, dtype=np.dtype([])) for arg in self.args]
                ).shape
        except ValueError as e:
            raise ValueError(f"Invalid arguments for cp.vstack: {e}") from e

    # All arguments must have the same width.
    def validate_arguments(self) -> None:
        self.shape_from_args()

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Stack the expressions vertically.

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
        return (lu.vstack(arg_objs, shape), [])

    def _verify_jacobian_args(self):
        return True

    def _jacobian(self):
        result = {}
        M = self.shape[0]

        row_offset = 0
        for arg in self.args:
            jac = arg.jacobian()
            m_j = arg.shape[0] if arg.ndim >= 2 else 1
            for k, (rows, cols, vals) in jac.items():
                new_rows = (rows % m_j) + row_offset + (rows // m_j) * M
                if k in result:
                    old_rows, old_cols, old_vals = result[k]
                    result[k] = (
                        np.concatenate([old_rows, new_rows]),
                        np.concatenate([old_cols, cols]),
                        np.concatenate([old_vals, vals]),
                    )
                else:
                    result[k] = (new_rows, cols, vals)
            row_offset += m_j

        return result

    def _verify_hess_vec_args(self):
        return True

    def _hess_vec(self, vec):
        M = self.shape[0]
        result = {}
        keys_require_summing = []

        row_offset = 0
        for arg in self.args:
            m_j = arg.shape[0] if arg.ndim >= 2 else 1

            arg_indices = np.arange(arg.size)
            output_indices = (arg_indices % m_j) + row_offset + (arg_indices // m_j) * M
            arg_vec = vec[output_indices]

            arg_result = arg.hess_vec(arg_vec)
            for k, v in arg_result.items():
                if k in result:
                    old_rows, old_cols, old_vals = result[k]
                    new_rows, new_cols, new_vals = v
                    result[k] = (
                        np.concatenate([old_rows, new_rows]),
                        np.concatenate([old_cols, new_cols]),
                        np.concatenate([old_vals, new_vals]),
                    )
                    keys_require_summing.append(k)
                else:
                    result[k] = v

            row_offset += m_j

        for k in set(keys_require_summing):
            rows, cols, vals = result[k]
            var1, var2 = k
            hess = coo_matrix((vals, (rows, cols)), shape=(var1.size, var2.size))
            hess.sum_duplicates()
            result[k] = (hess.row, hess.col, hess.data)

        return result
