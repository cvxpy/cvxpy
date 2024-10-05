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
from typing import List, Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.gmatmul import gmatmul
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression


class cumprod(AffAtom, AxisAtom):
    """
    Cumulative product of the elements of an expression.

    Attributes
    ----------
    expr : CVXPY expression
        The expression being multiplied.
    axis : int
        The axis to multiply across.
    """
    def __init__(self, expr: Expression, axis: int = 0) -> None:
        super(cumprod, self).__init__(expr, axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """
        Returns the cumulative product of the elements of the expression.
        """
        return np.cumprod(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
        """The same as the input."""
        return self.args[0].shape

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # TODO implement grad
        return []

    def get_data(self):
        """Returns the axis being summed."""
        return [self.axis]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """
        Cumulative product via difference matrix.

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
        n = arg_objs[0]
        A = lu.create_const(np.triu(np.ones(shape)), shape=shape)
        return (gmatmul(A, n), [])
