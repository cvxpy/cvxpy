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
from typing import Tuple

import numpy as np

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom


class trace(AffAtom):
    """The sum of the diagonal entries of a matrix.

    Parameters
    ----------
    expr : Expression
        The expression to sum the diagonal of.
    """

    def __init__(self, expr) -> None:
        super(trace, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Sums the diagonal entries.
        """
        return np.trace(values[0])

    def validate_arguments(self) -> None:
        """Checks that the argument is a square matrix.
        """
        shape = self.args[0].shape
        if self.args[0].ndim != 2 or shape[0] != shape[1]:
            raise ValueError("Argument to trace must be a square matrix.")

    def shape_from_args(self):
        """Always scalar.
        """
        return tuple()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None):
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
        return (lu.trace(arg_objs[0]), [])
