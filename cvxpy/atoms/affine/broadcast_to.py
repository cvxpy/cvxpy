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

from typing import List, Optional, Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


class broadcast_to(AffAtom):
    """Broadcast the expression given a shape input"""

    def __init__(self, expr, shape) -> None:
        self.broadcast_shape = shape
        self._shape = expr.shape
        super(broadcast_to, self).__init__(expr)

    def _supports_cpp(self) -> bool:
        return False

    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

    def numeric(self, values):
        return np.broadcast_to(values[0], shape=self.broadcast_shape)

    def get_data(self) -> List[Optional[int]]:
        return [self.broadcast_shape]

    def validate_arguments(self) -> None:
        np.broadcast_to(
            np.empty(self.shape, dtype=np.dtype([])),
            shape=self.broadcast_shape
        )

    def shape_from_args(self) -> Tuple[int, ...]:
        return self.broadcast_shape

    def graph_implementation(
        self,
        arg_objs,
        shape: Tuple[int, ...],
        data=None,
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Broadcast an expression to a given shape.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom. In this case data wraps axis

        Returns
        -------
        tuple
            (LinOp for the objective, list of constraints)
        """
        return (lu.broadcast_to(arg_objs, shape), [])
