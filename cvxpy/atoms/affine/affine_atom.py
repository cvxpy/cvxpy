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
import abc
from typing import Any, List, Tuple

import scipy.sparse as sp

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants import Constant
from cvxpy.utilities import performance_utils as perf


class AffAtom(Atom):
    """ Abstract base class for affine atoms. """
    __metaclass__ = abc.ABCMeta
    _allow_complex = True

    def sign_from_args(self) -> Tuple[bool, bool]:
        """By default, the sign is the most general of all the argument signs.
        """
        return u.sign.sum_signs([arg for arg in self.args])

    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        # Default is most generic argument.
        return all(arg.is_imag() for arg in self.args)

    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        # Default is most generic argument.
        return any(arg.is_complex() for arg in self.args)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        # Defaults to increasing.
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        # Defaults to increasing.
        return False

    def is_quadratic(self) -> bool:
        return all(arg.is_quadratic() for arg in self.args)

    def has_quadratic_term(self) -> bool:
        """Does the affine head of the expression contain a quadratic term?

        The affine head is all nodes with a path to the root node
        that does not pass through any non-affine atom. If the root node
        is non-affine, then the affine head is the root alone.
        """
        return any(arg.has_quadratic_term() for arg in self.args)

    def is_qpwa(self) -> bool:
        return all(arg.is_qpwa() for arg in self.args)

    def is_pwl(self) -> bool:
        return all(arg.is_pwl() for arg in self.args)

    # TODO is this right?
    @perf.compute_once
    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        for idx, arg in enumerate(self.args):
            if not ((self.is_incr(idx) and arg.is_psd()) or
                    (self.is_decr(idx) and arg.is_nsd())):
                return False
        return True

    @perf.compute_once
    def is_nsd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        for idx, arg in enumerate(self.args):
            if not ((self.is_decr(idx) and arg.is_psd()) or
                    (self.is_incr(idx) and arg.is_nsd())):
                return False
        return True

    def _grad(self, values) -> List[Any]:
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # TODO should be a simple function in cvxcore for this.
        # Make a fake lin op tree for the function.
        fake_args = []
        var_offsets = {}
        offset = 0
        for idx, arg in enumerate(self.args):
            if arg.is_constant():
                fake_args += [Constant(arg.value).canonical_form[0]]
            else:
                fake_args += [lu.create_var(arg.shape, idx)]
                var_offsets[idx] = offset
                offset += arg.size
        var_length = offset
        fake_expr, _ = self.graph_implementation(fake_args, self.shape,
                                                 self.get_data())
        param_to_size = {lo.CONSTANT_ID: 1}
        param_to_col = {lo.CONSTANT_ID: 0}
        # Get the matrix representation of the function.
        canon_mat = canonInterface.get_problem_matrix(
            [fake_expr],
            var_length,
            var_offsets,
            param_to_size,
            param_to_col,
            self.size,
        )
        # HACK TODO TODO convert tensors back to vectors.
        # COO = (V[lo.CONSTANT_ID][0], (J[lo.CONSTANT_ID][0], I[lo.CONSTANT_ID][0]))
        shape = (var_length + 1, self.size)
        stacked_grad = canon_mat.reshape(shape).tocsc()[:-1, :]
        # Break up into per argument matrices.
        grad_list = []
        start = 0
        for arg in self.args:
            if arg.is_constant():
                grad_shape = (arg.size, shape[1])
                if grad_shape == (1, 1):
                    grad_list += [0]
                else:
                    grad_list += [sp.coo_matrix(grad_shape, dtype='float64')]
            else:
                stop = start + arg.size
                grad_list += [stacked_grad[start:stop, :]]
                start = stop
        return grad_list
