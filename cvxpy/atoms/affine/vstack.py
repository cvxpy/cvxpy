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
            raise ValueError(f"Invalid shapes for vstack: {e}") from e

    # All arguments must have the same width.
    def validate_arguments(self) -> None:
        try:
            np.vstack(
                [np.empty(arg.shape, dtype=np.dtype([])) for arg in self.args]
                )
        except ValueError as e:
            raise ValueError(f"Invalid arguments for vstack: {e}") from e

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
