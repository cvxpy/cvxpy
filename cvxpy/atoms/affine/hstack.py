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


def hstack(arg_list) -> "Hstack":
    """Horizontal concatenation of an arbitrary number of Expressions.

    Parameters
    ----------
    arg_list : list of Expression
        The Expressions to concatenate.
    """
    arg_list = [AffAtom.cast_to_const(arg) for arg in arg_list]
    for idx, arg in enumerate(arg_list):
        if arg.ndim == 0:
            arg_list[idx] = arg.flatten(order='F')
    return Hstack(*arg_list)


class Hstack(AffAtom):
    """ Horizontal concatenation """
    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    # Returns the hstack of the values.
    def numeric(self, values):
        return np.hstack(values)

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self) -> Tuple[int, ...]:
        try:
            return np.hstack(
                [np.empty(arg.shape, dtype=np.dtype([])) for arg in self.args]
                ).shape
        except ValueError as e:
            raise ValueError(f"Invalid arguments for cp.hstack: {e}") from e

    # All arguments must have the same width.
    def validate_arguments(self) -> None:
        self.shape_from_args()

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Stack the expressions horizontally.

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
        return (lu.hstack(arg_objs, shape), [])

    def _verify_jacobian_args(self):
        return True

    def _jacobian(self):
        result = {}

        flat_offset = 0
        for arg in self.args:
            jac = arg.jacobian()

            for k, (rows, cols, vals) in jac.items():
                new_rows = rows + flat_offset
                if k in result:
                    old_rows, old_cols, old_vals = result[k]
                    result[k] = (
                        np.concatenate([old_rows, new_rows]),
                        np.concatenate([old_cols, cols]),
                        np.concatenate([old_vals, vals]),
                    )
                else:
                    result[k] = (new_rows, cols, vals)

            flat_offset += arg.size

        return result

    def _verify_hess_vec_args(self):
        return True

    def _hess_vec(self, vec):
        result = {}
        keys_require_summing = []

        flat_offset = 0
        for arg in self.args:
            arg_vec = vec[flat_offset:flat_offset + arg.size]

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

            flat_offset += arg.size

        for k in set(keys_require_summing):
            rows, cols, vals = result[k]
            var1, var2 = k
            hess = coo_matrix((vals, (rows, cols)), shape=(var1.size, var2.size))
            hess.sum_duplicates()
            result[k] = (hess.row, hess.col, hess.data)

        return result
