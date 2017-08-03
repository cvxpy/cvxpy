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
import numpy.linalg as LA
import scipy.sparse as sp


class Leaf(expression.Expression):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, shape, value=None, nonneg=False, nonpos=False,
                 real=True, imag=False,
                 symmetric=False, diag=False, PSD=False,
                 NSD=False, Hermitian=False,
                 boolean=False, integer=False,
                 sparsity=None):
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
          boolean: Is the variable boolean? Either True, which constrains
                   the entire Variable to be boolean, False, or a list of
                   indices which should be constrained as boolean, where each
                   index is a tuple of length exactly equal to the
                   length of shape.
          integer: Is the variable integer? The semantics are the same as the
                   boolean argument.
          sparsity: Fixed sparsity pattern for the variable.
        """
        # TODO remove after adding 0D and 1D support.
        if isinstance(shape, int):
            shape = (shape, 1)
        elif len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = (shape[0], 1)
        self._shape = tuple(int(d) for d in shape)

        if (PSD or NSD or symmetric or diag) and shape[0] != shape[1]:
            raise ValueError("Invalid dimensions %s. Must be a square matrix." % (shape,))

        # Process attributes.
        self.attributes = {'nonneg': nonneg, 'nonpos': nonpos,
                           'real': real, 'imag': imag,
                           'symmetric': symmetric, 'diag': diag,
                           'PSD': PSD, 'NSD': NSD,
                           'Hermitian': Hermitian, 'boolean': bool(boolean),
                           'integer':  integer, 'sparsity': sparsity}

        if boolean:
            self.boolean_idx = boolean if not type(boolean) == bool else list(
                np.ndindex(shape))
        else:
            self.boolean_idx = []

        if integer:
            self.integer_idx = integer if not type(integer) == bool else list(
                np.ndindex(shape))
        else:
            self.integer_idx = []

        # Only one attribute besides real can be True (except can be nonneg and
        # nonpos, similarly for boolean and integer).
        true_attr = sum([1 for k, v in self.attributes.items() if k != 'real' and v])
        if nonneg and nonpos:
            true_attr -= 1
        if boolean and integer:
            true_attr -= 1
        if true_attr > 1:
            raise ValueError("Cannot set more than one special attribute in %s."
                             % self.__class__.__name__)

        if value is not None:
            self.value = value

        self.args = []

    def _get_attr_str(self):
        """Get a string representing the attributes.
        """
        attr_str = ""
        for attr, val in self.attributes.items():
            if attr != 'real' and val:
                attr_str += ", %s=%s" % (attr, val)
        return attr_str

    def copy(self, args=None, id_objects={}):
        """Returns a shallow copy of the object.

        Used to reconstruct an object tree.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the object. If args=None, use the
            current args of the object.

        Returns
        -------
        Expression
        """
        if id(self) in id_objects:
            return id_objects[id(self)]
        return self  # Leaves are not deep copied.

    def get_data(self):
        """Leaves are not copied.
        """
        pass

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
        return self.attributes['nonneg'] or self.attributes['boolean']

    def is_nonpos(self):
        """Is the expression nonpositive?
        """
        return self.attributes['nonpos']

    def is_symmetric(self):
        """Is the Leaf symmetric.
        """
        return self.is_scalar() or \
            any([self.attributes[key] for key in ['diag', 'symmetric', 'PSD', 'NSD']])

    @property
    def domain(self):
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        # Default is full domain.
        return []

    def round(self, val):
        """Project value onto the attribute set of the leaf.
        Parameters
        ----------
        val : numeric type
            The value assigned.

        Returns
        -------
        numeric type
            The value converted to the proper matrix type.
        """
        # Only one attribute can be active at once (besides real).
        if self.attributes['nonpos']:
            return np.minimum(val, 0.)
        elif self.attributes['nonneg']:
            return np.maximum(val, 0.)
        elif self.attributes['boolean']:
            return np.round(np.clip(val, 0., 1.))
        elif self.attributes['integer']:
            return np.round(val)
        elif self.attributes['diag']:
            return sp.diags([np.diag(val)], [0])
        elif any([self.attributes[key] for
                  key in ['symmetric', 'PSD', 'NSD']]):
            val = (val + val.T)/2
            if self.attributes['symmetric']:
                return val
            w, V = LA.eig(val)
            if self.attributes['PSD']:
                w = np.maximum(w, 0)
            else:  # NSD
                w = np.minimum(w, 0)
            return V.dot(np.diag(w)).dot(V.T)
        else:
            return val

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
            # Convert val to ndarray.
            val = intf.DEFAULT_INTF.const_to_matrix(val)
            if val.shape != self.shape:
                raise ValueError(
                    "Invalid dimensions %s for %s value." %
                    (val.shape, self.__class__.__name__)
                )
            elif np.any(self.round(val) != val):
                if self.attributes['nonneg']:
                    attr_str = 'nonnegative'
                elif self.attributes['nonpos']:
                    attr_str = 'nonpositive'
                elif self.attributes['diag']:
                    attr_str = 'diagonal'
                elif self.attributes['PSD']:
                    attr_str = 'positive semidefinite'
                elif self.attributes['NSD']:
                    attr_str = 'negative semidefinite'
                else:
                    attr_str = [k for (k, v) in self.attributes.items() if v and k != 'real'][0]
                raise ValueError(
                    "%s value must be %s." % (self.__class__.__name__, attr_str)
                )
        return val

    def is_quadratic(self):
        """Leaf nodes are always quadratic.
        """
        return True

    def is_pwl(self):
        """Leaf nodes are always piecewise linear.
        """
        return True

    def atoms(self):
        return []
