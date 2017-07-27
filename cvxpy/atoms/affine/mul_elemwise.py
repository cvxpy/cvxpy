"""
Copyright 2017 Steven Diamond

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


class mul_elemwise(AffAtom):
    """ Multiplies two expressions elementwise.

    The first expression must be constant.
    """

    def __init__(self, lh_const, rh_expr):
        super(mul_elemwise, self).__init__(lh_const, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Multiplies the values elementwise.
        """
        return np.multiply(values[0], values[1])

    def validate_arguments(self):
        """Checks that the arguments are valid.

           Left-hand argument must be constant.
        """
        if not self.args[0].is_constant():
            raise ValueError(("The first argument to mul_elemwise must "
                              "be constant."))

    def size_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        return u.shape.sum_shapes([arg.size for arg in self.args])

    def sign_from_args(self):
        """Same as times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_negative()

    def is_quadratic(self):
        """Quadratic if x is quadratic.
        """
        return self.args[1].is_quadratic()

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Multiply the expressions elementwise.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        # One of the arguments is a scalar, so we can use normal multiplication.
        if arg_objs[0].size != arg_objs[1].size:
            return (lu.mul_expr(arg_objs[0], arg_objs[1], size), [])
        else:
            return (lu.mul_elemwise(arg_objs[0], arg_objs[1]), [])
