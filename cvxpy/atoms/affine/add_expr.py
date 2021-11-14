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
import operator as op
from functools import reduce
from typing import List, Tuple

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint


class AddExpression(AffAtom):
    """The sum of any number of expressions.
    """

    def __init__(self, arg_groups) -> None:
        # For efficiency group args as sums.
        self._arg_groups = arg_groups
        super(AddExpression, self).__init__(*arg_groups)
        self.args = []
        for group in arg_groups:
            self.args += self.expand_args(group)

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def expand_args(self, expr):
        """Helper function to extract the arguments from an AddExpression.
        """
        if isinstance(expr, AddExpression):
            return expr.args
        else:
            return [expr]

    def name(self) -> str:
        result = str(self.args[0])
        for i in range(1, len(self.args)):
            result += " + " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.add, values)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        symm_args = all(arg.is_symmetric() for arg in self.args)
        return self.shape[0] == self.shape[1] and symm_args

    def is_hermitian(self) -> bool:
        """Is the expression Hermitian?
        """
        herm_args = all(arg.is_hermitian() for arg in self.args)
        return self.shape[0] == self.shape[1] and herm_args

    # As __init__ takes in the arg_groups instead of args, we need a special
    # copy() function.
    def copy(self, args=None, id_objects=None):
        """Returns a shallow copy of the AddExpression atom.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the atom. If args=None, use the
            current args of the atom.

        Returns
        -------
        AddExpression atom
        """
        if args is None:
            args = self._arg_groups
        # Takes advantage of _arg_groups if present for efficiency.
        copy = type(self).__new__(type(self))
        copy.__init__(args)
        return copy

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Sum the linear expressions.

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
        for i, arg in enumerate(arg_objs):
            if arg.shape != shape and lu.is_scalar(arg):
                arg_objs[i] = lu.promote(arg, shape)
        return (lu.sum_expr(arg_objs), [])
