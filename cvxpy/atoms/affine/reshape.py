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
import numbers
from typing import List, Tuple

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression


class reshape(AffAtom):
    """Reshapes the expression.

    Vectorizes the expression then unvectorizes it into the new shape.
    The entries are reshaped and stored in column-major order, also known
    as Fortran order.

    Parameters
    ----------
    expr : Expression
       The expression to promote.
    shape : tuple or int
        The shape to promote to.
    order : F(ortran) or C
    """

    def __init__(self, expr, shape: Tuple[int, int], order: str = 'F') -> None:
        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)
        if len(shape) > 2:
            raise ValueError("Expressions of dimension greater than 2 "
                             "are not supported.")
        self._shape = tuple(shape)
        assert order in ['F', 'C']
        self.order = order
        super(reshape, self).__init__(expr)

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
        return np.reshape(values[0], self.shape, self.order)

    def validate_arguments(self) -> None:
        """Checks that the new shape has the same number of entries as the old.
        """
        old_len = self.args[0].size
        new_len = np.prod(self._shape, dtype=int)
        if not old_len == new_len:
            raise ValueError(
                "Invalid reshape dimensions %s." % (self._shape,)
            )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape from the rows, cols arguments.
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
                result = lu.reshape(arg, (shape[1], shape[0]))
                return (lu.transpose(result), [])


def deep_flatten(x):
    # base cases
    if isinstance(x, Expression):
        if len(x.shape) == 1:
            return x
        else:
            return x.flatten()
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float)):
        x = Expression.cast_to_const(x)
        return x.flatten()
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
