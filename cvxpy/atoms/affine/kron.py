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
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class kron(AffAtom):
    """Kronecker product.
    """
    # TODO work with right hand constant.
    # TODO(akshayka): make DGP-compatible

    def __init__(self, lh_expr, rh_expr):
        super(kron, self).__init__(lh_expr, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Kronecker product of the two values.
        """
        return np.kron(values[0], values[1])

    def validate_arguments(self):
        """Checks that both arguments are vectors, and the first is constant.
        """
        if not self.args[0].is_constant():
            raise ValueError("The first argument to kron must be constant.")
        elif self.args[0].ndim != 2 or self.args[1].ndim != 2:
            raise ValueError("kron requires matrix arguments.")

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        rows = self.args[0].shape[0]*self.args[1].shape[0]
        cols = self.args[0].shape[1]*self.args[1].shape[1]
        return (rows, cols)

    def sign_from_args(self):
        """Same as times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Kronecker product of two matrices.

        Parameters
        ----------
        arg_objs : list
            LinOp for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.kron(arg_objs[0], arg_objs[1], shape), [])
