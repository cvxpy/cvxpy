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
from __future__ import annotations

import numbers
import warnings
from typing import List, Literal, Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.settings as s
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import DEFAULT_ORDER_DEPRECATION_MSG, Expression
from cvxpy.utilities.shape import size_from_shape


class reshape(AffAtom):
    """
    Reshapes the expression.

    Vectorizes the expression then unvectorizes it into the new shape.
    The entries are reshaped and stored in column-major order, also known
    as Fortran order.

    Parameters
    ----------
    expr : Expression
       The expression to reshape
    shape : tuple or int
        The shape to reshape to
    order : F(ortran) or C
    """

    def __init__(
        self,
        expr,
        shape: int | Tuple[int, ...],
        order: Literal["F", "C", None] = None
    ) -> None:
        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)
        if not s.ALLOW_ND_EXPR and len(shape) > 2:
            raise ValueError("Expressions of dimension greater than 2 "
                             "are not supported.")
        if any(d == -1 for d in shape):
            shape = self._infer_shape(shape, expr.size)

        self._shape = tuple(shape)
        if order is None:
            reshape_order_warning = DEFAULT_ORDER_DEPRECATION_MSG.replace("FUNC_NAME", "reshape")
            warnings.warn(reshape_order_warning, FutureWarning)
            order = 'F'
        assert order in ['F', 'C']
        self.order = order
        super(reshape, self).__init__(expr)

    @staticmethod
    def _infer_shape(shape: Tuple[int, ...], size: int) -> Tuple[int, ...]:
        assert shape.count(-1) == 1, "Only one dimension can be -1."
        if len(shape) == 1:
            shape = (size,)
        else:
            unspecified_index = shape.index(-1)
            specified = shape[1 - unspecified_index]
            assert specified >= 0, "Specified dimension must be nonnegative."
            unspecified, remainder = np.divmod(size, shape[1 - unspecified_index])
            if remainder != 0:
                raise ValueError(
                    f"Cannot reshape expression of size {size} into shape {shape}."
                )
            shape = tuple(unspecified if d == -1 else specified for d in shape)
        return shape

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Reshape the value.
        """
        return np.reshape(values[0], self.shape, order=self.order)

    def validate_arguments(self) -> None:
        """Checks that the new shape has the same number of entries as the old.
        """
        old_len = self.args[0].size
        new_len = size_from_shape(self._shape)
        if not old_len == new_len:
            raise ValueError(
                "Invalid reshape dimensions %s." % (self._shape,)
            )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape argument.
        """
        return self._shape

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._shape, self.order]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Reshape

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
        arg = arg_objs[0]
        if data[1] == 'F':
            return (lu.reshape(arg, shape), [])
        else:  # 'C':
            arg = lu.transpose(arg)
            if len(shape) <= 1:
                return (lu.reshape(arg, shape), [])
            else:
                result = lu.reshape(arg, shape[::-1])
                return (lu.transpose(result), [])


def deep_flatten(x):
    # base cases
    if isinstance(x, Expression):
        if len(x.shape) == 1:
            return x
        else:
            return x.flatten(order='F')
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float)):
        x = Expression.cast_to_const(x)
        return x.flatten(order='F')
    # recursion
    if isinstance(x, list):
        y = []
        for x0 in x:
            x1 = deep_flatten(x0)
            y.append(x1)
        y = hstack(y)
        return y
    msg = 'The input to deep_flatten must be an Expression, a NumPy array, an int'\
          + ' or float, or a nested list thereof. Received input of type %s' % type(x)
    raise ValueError(msg)
