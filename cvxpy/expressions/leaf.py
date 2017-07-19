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
from cvxpy.expressions import expression
import cvxpy.interface as intf
import numpy as np


class Leaf(expression.Expression):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, shape, value=None, nonneg=False, nonpos=False,
                 real=True, imag=False,
                 symmetric=False, diag=False, PSD=False,
                 NSD=False, Hermitian=False,
                 boolean=False, integer=False, sparsity=None):
        """
        Args:
          shape: The leaf dimensions.
          value: A value to assign to the leaf.
          nonneg: Is the variable constrained to be nonnegative?
          nonpos: Is the variable constrained to be nonpositive?
          real: Does the variable have a real part?
          imag: Does the variable have an imaginary part?
          symmetric: Is the variable symmetric?
          diag: Is the variable diagonal?
          PSD: Is the variable constrained to be positive semidefinite?
          NSD: Is the variable constrained to be negative semidefinite?
          Hermitian: Is the variable Hermitian?
          boolean: Is the variable boolean?
          integer: Is the variable integer?
          sparsity: Fixed sparsity pattern for the variable.
        """
        # TODO remove after adding 0D and 1D support.
        if isinstance(shape, int):
            shape = (shape, 1)
        elif len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = (shape[0], 1)
        self._shape = shape

        # Process attributes.
        self.attributes = {'nonneg': nonneg, 'nonpos': nonpos,
                           'real': real, 'imag': imag,
                           'symmetric': symmetric, 'diag': diag,
                           'PSD': PSD, 'NSD': NSD,
                           'Hermitian': Hermitian, 'boolean': boolean,
                           'integer':  integer, 'sparsity': sparsity}
        # Only one attribute besides real can be True (except can be nonneg and nonpos).
        true_attr = sum([1 for k, v in self.attributes.items() if k != 'real' and v])
        if nonneg and nonpos:
            true_attr -= 1
        if true_attr > 1:
            raise ValueError("Cannot set more than one special attribute in %s." % self.__class__)

        if value is not None:
            self.value = value

        self.args = []

    @property
    def shape(self):
        """Returns the dimensions of the expression.
        """
        return self._shape

    def variables(self):
        """Default is empty list of Variables.
        """
        return []

    def parameters(self):
        """Default is empty list of Parameters.
        """
        return []

    def constants(self):
        """Default is empty list of Constants.
        """
        return []

    def is_convex(self):
        """Is the expression convex?
        """
        return True

    def is_concave(self):
        """Is the expression concave?
        """
        return True

    def is_nonneg(self):
        """Is the expression nonnegative?
        """
        return self.attributes['nonneg']

    def is_nonpos(self):
        """Is the expression nonpositive?
        """
        return self.attributes['nonpos']

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        # Default is full domain.
        return []

    def _validate_value(self, val):
        """Check that the value satisfies the leaf's symbolic attributes.

        Parameters
        ----------
        val : numeric type
            The value assigned.

        Returns
        -------
        numeric type
            The value converted to the proper matrix type.
        """
        if val is not None:
            # Convert val to the proper matrix type.
            val = intf.DEFAULT_INTF.const_to_matrix(val)
            shape = intf.shape(val)
            if shape != self.shape:
                raise ValueError(
                    "Invalid dimensions (%s, %s) for %s value." %
                    (shape[0], shape[1], self.__class__.__name__)
                )
            # All signs are valid if sign is unknown.
            # Otherwise value sign must match declared sign.
            pos_val, neg_val = intf.sign(val)
            if self.is_nonneg() and not pos_val or \
               self.is_nonpos() and not neg_val:
                raise ValueError(
                    "Invalid sign for %s value." % self.__class__.__name__
                )
            # Round to correct sign.
            elif self.is_nonneg():
                val = np.maximum(val, 0)
            elif self.is_nonpos():
                val = np.minimum(val, 0)
        return val

    def is_quadratic(self):
        """Leaf nodes are always quadratic.
        """
        return True

    def is_pwl(self):
        """Leaf nodes are always piecewise linear.
        """
        return True
