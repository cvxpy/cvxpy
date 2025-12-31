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

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy as scipy
import scipy.sparse as sp

import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants.parameter import is_param_free


class quad_over_lin(AxisAtom):
    """:math:`(sum_{ij}X^2_{ij})/y`

    When axis is specified, computes the sum of squares along that axis,
    returning a vector instead of a scalar.
    """
    _allow_complex = True

    def __init__(
        self,
        x,
        y,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> None:
        self.axis = axis
        self.keepdims = keepdims
        # Call Atom.__init__ directly since we have two args
        Atom.__init__(self, x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        x_val = values[0]
        y_val = values[1].item()
        if self.args[0].is_complex():
            squared = np.square(x_val.imag) + np.square(x_val.real)
        else:
            squared = np.square(x_val)

        return squared.sum(axis=self.axis, keepdims=self.keepdims) / y_val

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        # y > 0.
        return [self.args[1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # Gradient not implemented for axis case
        if self.axis is not None:
            return [None, None]

        X = values[0]
        y = values[1]
        if y <= 0:
            return [None, None]
        else:
            # DX = 2X/y, Dy = -||X||^2_2/y^2
            if self.args[0].is_complex():
                Dy = -(np.square(X.real) + np.square(X.imag)).sum()/np.square(y)
            else:
                Dy = -np.square(X).sum()/np.square(y)

            # Ensure Dy is a scalar for proper sparse array construction
            Dy = float(np.asarray(Dy).item() if np.asarray(Dy).ndim > 0 else Dy)
            Dy = sp.csc_array([[Dy]])
            DX = 2.0*X/y
            # Use F-order to match CVXPY's vectorization convention
            DX = np.reshape(DX, (self.args[0].size, 1), order='F')
            DX = scipy.sparse.csc_array(DX)
            return [DX, Dy]

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        # Disable DPP when the second argument is a parameter.
        if u.scopes.dpp_scope_active():
            return is_param_free(self.args[1])
        else:
            return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return (idx == 0) and self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return ((idx == 0) and self.args[idx].is_nonpos()) or (idx == 1)

    def validate_arguments(self) -> None:
        """Check dimensions of arguments.
        """
        if not self.args[1].is_scalar():
            raise ValueError("The second argument to quad_over_lin must be a scalar.")
        if self.args[1].is_complex():
            raise ValueError("The second argument to quad_over_lin cannot be complex.")
        # AxisAtom.validate_arguments handles axis validation
        super(quad_over_lin, self).validate_arguments()

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine and y is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        """A quadratic term if y is constant.
        """
        return self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        """Quadratic of piecewise affine if x is PWL and y is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()
