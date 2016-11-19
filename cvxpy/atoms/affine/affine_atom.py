"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import abc
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants import Constant
import canonInterface
import scipy.sparse as sp


class AffAtom(Atom):
    """ Abstract base class for affine atoms. """
    __metaclass__ = abc.ABCMeta

    def sign_from_args(self):
        """By default, the sign is the most general of all the argument signs.
        """
        return u.sign.sum_signs([arg for arg in self.args])

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        # Defaults to increasing.
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        # Defaults to increasing.
        return False

    def is_quadratic(self):
        return all([arg.is_quadratic() for arg in self.args])

    def is_pwl(self):
        return all([arg.is_pwl() for arg in self.args])

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # TODO should be a simple function in CVXcanon for this.
        # Make a fake lin op tree for the function.
        fake_args = []
        var_offsets = {}
        offset = 0
        for idx, arg in enumerate(self.args):
            if arg.is_constant():
                fake_args += [Constant(arg.value).canonical_form[0]]
            else:
                fake_args += [lu.create_var(arg.size, idx)]
                var_offsets[idx] = offset
                offset += arg.size[0]*arg.size[1]
        fake_expr, _ = self.graph_implementation(fake_args, self.size,
                                                 self.get_data())
        # Get the matrix representation of the function.
        V, I, J, _ = canonInterface.get_problem_matrix(
            [lu.create_eq(fake_expr)],
            var_offsets,
            None
        )
        shape = (offset, self.size[0]*self.size[1])
        stacked_grad = sp.coo_matrix((V, (J, I)), shape=shape).tocsc()
        # Break up into per argument matrices.
        grad_list = []
        start = 0
        for arg in self.args:
            if arg.is_constant():
                grad_shape = (arg.size[0]*arg.size[1], shape[1])
                if grad_shape == (1, 1):
                    grad_list += [0]
                else:
                    grad_list += [sp.coo_matrix(grad_shape, dtype='float64')]
            else:
                stop = start + arg.size[0]*arg.size[1]
                grad_list += [stacked_grad[start:stop, :]]
                start = stop
        return grad_list
