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

from cvxpy.atoms.affine.affine_atom import AffAtom
from typing import Tuple

import cvxpy.utilities as u
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class conv(AffAtom):
    """ 1D discrete convolution of two vectors.

    The discrete convolution :math:`c` of vectors :math:`a` and :math:`b` of
    lengths :math:`n` and :math:`m`, respectively, is a length-:math:`(n+m-1)`
    vector where

    .. math::

        c_k = \\sum_{i+j=k} a_ib_j, \\quad k=0, \\ldots, n+m-2.

    Parameters
    ----------
    lh_expr : Constant
        A constant 1D vector or a 2D column vector.
    rh_expr : Expression
        A 1D vector or a 2D column vector.
    """
    # TODO work with right hand constant.
    # TODO(akshayka): make DGP-compatible

    def __init__(self, lh_expr, rh_expr) -> None:
        super(conv, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convolve the two values.
        """
        # Convert values to 1D.
        values = list(map(intf.from_2D_to_1D, values))
        return np.convolve(values[0], values[1])

    def validate_arguments(self) -> None:
        """Checks that both arguments are vectors, and the first is constant.
        """
        if not self.args[0].is_vector() or not self.args[1].is_vector():
            raise ValueError("The arguments to conv must resolve to vectors.")
        if not self.args[0].is_constant():
            raise ValueError("The first argument to conv must be constant.")

    def shape_from_args(self) -> Tuple[int, int]:
        """The sum of the argument dimensions - 1.
        """
        lh_length = self.args[0].shape[0]
        rh_length = self.args[1].shape[0]
        return (lh_length + rh_length - 1, 1)

    def sign_from_args(self):
        """Same as times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    def graph_implementation(self, arg_objs, shape, data=None):
        """Convolve two vectors.

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
        return (lu.conv(arg_objs[0], arg_objs[1], shape), [])
