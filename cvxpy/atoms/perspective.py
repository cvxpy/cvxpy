"""
Copyright 2019 Shane Barratt

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

from cvxpy.atoms.atom import Atom
import numpy as np
import scipy.sparse as sp
import scipy as scipy
import IPython as ipy

class perspective(Atom):
    """ :math:`\text{perspective}(f, x, t) = tf(x/t)` """

    def __init__(self, *args, atom=None):
        self._atom = atom
        self._atom_initialized = atom(*args[:-1])
        super(perspective, self).__init__(*args)
    
    def get_data(self):
        return [self._atom, self._atom_initialized]
    
    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the evaluation of the perspective.
        """
        args = values[:-1]
        t = values[-1].flatten()[0]

        if t > 0:
            args = [a / t for a in args]
            return t * self._atom(*args).value # we have to do this
        elif all([np.all(a == 0) for a in args]) and t == 0:
            return 0.0
        else:
            return float("inf")

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[-1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return NotImplementedError

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return self._atom_initialized.sign_from_args()

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return self._atom_initialized.is_atom_convex()

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return self._atom_initialized.is_atom_concave()

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return False

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        if idx < len(self.args) - 1:
            return self._atom_initialized.is_incr(idx)
        else:
            return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        if idx < len(self.args) - 1:
            return self._atom_initialized.is_decr(idx)
        else:
            return False

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        self._atom_initialized.validate_arguments()
        if not self.args[-1].is_scalar():
            raise ValueError("The last argument to perspective must be a scalar.")
        if self.args[-1].is_complex():
            raise ValueError("The last argument to perspective cannot be complex.")
        super(perspective, self).validate_arguments()

    def is_quadratic(self):
        """Quadratic if XXX.
        """
        return False

    def is_qpwa(self):
        """Quadratic of piecewise affine if XXX.
        """
        return False
    
