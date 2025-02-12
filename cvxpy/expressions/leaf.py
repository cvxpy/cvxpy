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
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from cvxpy import Constant, Parameter, Variable
    from cvxpy.atoms.atom import Atom

import numbers
import warnings

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import expression
from cvxpy.settings import (
    GENERAL_PROJECTION_TOL,
    PSD_NSD_PROJECTION_TOL,
    SPARSE_PROJECTION_TOL,
)
from cvxpy.utilities.coo_array_compat import get_coords


class Leaf(expression.Expression):
    """
    A leaf node of an expression tree; i.e., a Variable, Constant, or Parameter.

    A leaf may carry *attributes* that constrain the set values permissible
    for it. Leafs can have no more than one attribute, with the exception
    that a leaf may be both ``nonpos`` and ``nonneg`` or both ``boolean``
    in some indices and ``integer`` in others.

    An error is raised if a leaf is assigned a value that contradicts
    one or more of its attributes. See the ``project`` method for a convenient
    way to project a value onto a leaf's domain.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the leaf, e.g., ``(3, 2)`` or ``2``.
    value : numeric type
        A value to assign to the leaf.
    nonneg : bool
        Is the variable constrained to be nonnegative?
    nonpos : bool
        Is the variable constrained to be nonpositive?
    complex : bool
        Is the variable complex valued?
    symmetric : bool
        Is the variable symmetric?
    diag : bool
        Is the variable diagonal?
    PSD : bool
        Is the variable constrained to be positive semidefinite?
    NSD : bool
        Is the variable constrained to be negative semidefinite?
    Hermitian : bool
        Is the variable Hermitian?
    boolean : bool or Iterable
        Is the variable boolean? True, which constrains
        the entire Variable to be boolean, False, or a list of
        indices which should be constrained as boolean, where each
        index is a tuple of length exactly equal to the
        length of shape.
    integer : bool or Iterable
        Is the variable integer? The semantics are the same as the
        boolean argument.
    sparsity : Iterable
        Is the variable sparse?
    pos : bool
        Is the variable positive?
    neg : bool
        Is the variable negative?
    bounds : Iterable
        An iterable of length two specifying lower and upper bounds.
    """

    def __init__(
        self, shape: int | tuple[int, ...], value = None, nonneg: bool = False,
        nonpos: bool = False, complex: bool = False, imag: bool = False,
        symmetric: bool = False, diag: bool = False, PSD: bool = False,
        NSD: bool = False, hermitian: bool = False,
        boolean: Iterable | bool = False, integer: Iterable | bool = False,
        sparsity: Iterable | bool = False, pos: bool = False, neg: bool = False,
        bounds: Iterable | None = None
    ) -> None:
        if isinstance(shape, numbers.Integral):
            shape = (int(shape),)
        elif not s.ALLOW_ND_EXPR and len(shape) > 2:
            raise ValueError("Expressions of dimension greater than 2 "
                             "are not supported.")
        for d in shape:
            if not isinstance(d, numbers.Integral) or d <= 0:
                raise ValueError("Invalid dimensions %s." % (shape,))
        shape = tuple(shape)
        self._shape = shape

        if (PSD or NSD or symmetric or diag or hermitian) and (len(shape) != 2
                                                               or shape[0] != shape[1]):
            raise ValueError("Invalid dimensions %s. Must be a square matrix."
                             % (shape,))

        # Process attributes.
        self.attributes = {'nonneg': nonneg, 'nonpos': nonpos,
                           'pos': pos, 'neg': neg,
                           'complex': complex, 'imag': imag,
                           'symmetric': symmetric, 'diag': diag,
                           'PSD': PSD, 'NSD': NSD,
                           'hermitian': hermitian, 'boolean': boolean,
                           'integer':  integer, 'sparsity': sparsity, 'bounds': bounds}
        if boolean is True:
            shape = max(shape, (1,))
            flat_idx = np.arange(np.prod(shape))
            self.boolean_idx = np.unravel_index(flat_idx, shape, order='F')
        elif boolean is False:
            self.boolean_idx = []
        else:
            self.boolean_idx = boolean
        if integer is True:
            shape = max(shape, (1,))
            flat_idx = np.arange(np.prod(shape))
            self.integer_idx = np.unravel_index(flat_idx, shape, order='F')
        elif integer is False:
            self.integer_idx = []
        else:
            self.integer_idx = integer
        if sparsity:
            self.sparse_idx = self._validate_indices(sparsity)
        else:
            self.sparse_idx = None
        # Only one attribute be True (except can be boolean and integer).
        true_attr = sum(1 for k, v in self.attributes.items() if v)
        # HACK we should remove this feature or allow multiple attributes in general.
        if boolean and integer:
            true_attr -= 1
        if true_attr > 1:
            raise ValueError("Cannot set more than one special attribute in %s."
                             % self.__class__.__name__)
        if value is not None:
            self.value = value

        self.args = []
        self.bounds = self._ensure_valid_bounds(bounds)

    def _validate_indices(self, indices: list[tuple[int]] | tuple[np.ndarray]) -> tuple[np.ndarray]:
        """
        Validate the sparsity pattern for a leaf node.

        Parameters:
        indices: List or tuple of indices indicating the positions of non-zero elements.
        """
        if self._shape == ():
            if indices != []:
                raise ValueError("Indices should have 0 dimensions.")
            return []
        # Attempt to form a COO_array with the indices matrix provided;
        # this will raise errors if invalid.
        validator = sp.coo_array((np.empty(len(indices[0])), indices), shape=self._shape)
        # Apply an in-place transformation to the coordinates to reduce the
        # validator to canonical form
        validator.sum_duplicates()
        # Return the canonicalized coordinates
        return get_coords(validator)

    def _get_attr_str(self) -> str:
        """Get a string representing the attributes."""
        attr_str = ""
        for attr, val in self.attributes.items():
            if attr != 'real' and val:
                attr_str += ", %s=%s" % (attr, val)
        return attr_str

    def copy(self, args=None, id_objects=None):
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
        id_objects = {} if id_objects is None else id_objects
        if id(self) in id_objects:
            return id_objects[id(self)]
        return self  # Leaves are not deep copied.

    def get_data(self) -> None:
        """Leaves are not copied."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The dimensions of the expression."""
        return self._shape

    def variables(self) -> list[Variable]:
        """Default is empty list of Variables."""
        return []

    def parameters(self) -> list[Parameter]:
        """Default is empty list of Parameters."""
        return []

    def constants(self) -> list[Constant]:
        """Default is empty list of Constants."""
        return []

    def is_convex(self) -> bool:
        """Is the expression convex?"""
        return True

    def is_concave(self) -> bool:
        """Is the expression concave?"""
        return True

    def is_log_log_convex(self) -> bool:
        """Is the expression log-log convex?"""
        return self.is_pos()

    def is_log_log_concave(self) -> bool:
        """Is the expression log-log concave?"""
        return self.is_pos()

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?"""
        return (self.attributes['nonneg'] or self.attributes['pos'] or
                self.attributes['boolean'])

    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?"""
        return self.attributes['nonpos'] or self.attributes['neg']

    def is_pos(self) -> bool:
        """Is the expression positive?"""
        return self.attributes['pos']

    def is_neg(self) -> bool:
        """Is the expression negative?"""
        return self.attributes['neg']

    def is_hermitian(self) -> bool:
        """Is the Leaf hermitian?"""
        return (self.is_real() and self.is_symmetric()) or \
            self.attributes['hermitian'] or self.is_psd() or self.is_nsd()

    def is_symmetric(self) -> bool:
        """Is the Leaf symmetric?"""
        return self.is_scalar() or \
            any(self.attributes[key] for key in ['diag', 'symmetric', 'PSD', 'NSD'])

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?"""
        return self.attributes['imag']

    def is_complex(self) -> bool:
        """Is the Leaf complex valued?"""
        return self.attributes['complex'] or self.is_imag() or self.attributes['hermitian']

    def _has_lower_bounds(self) -> bool:
        """Does the variable have lower bounds?"""
        if self.is_nonneg():
            return True
        elif self.attributes['bounds'] is not None:
            lower_bound = self.attributes['bounds'][0]
            if np.isscalar(lower_bound):
                return lower_bound != -np.inf
            else:
                return np.any(lower_bound != -np.inf)
        else:
            return False

    def _has_upper_bounds(self) -> bool:
        """Does the variable have upper bounds?"""
        if self.is_nonpos():
            return True
        elif self.attributes['bounds'] is not None:
            upper_bound = self.attributes['bounds'][1]
            if np.isscalar(upper_bound):
                return upper_bound != np.inf
            else:
                return np.any(upper_bound != np.inf)
        else:
            return False

    def _bound_domain(self, term: expression.Expression, constraints: list[Constraint]) -> None:
        """A utility function to append constraints from lower and upper bounds.

        Parameters
        ----------
        term: The term to encode in the constraints.
        constraints: An existing list of constraitns to append to.
        """
        if self.attributes['nonneg'] or self.attributes['pos']:
            constraints.append(term >= 0)
        elif self.attributes['nonpos'] or self.attributes['neg']:
            constraints.append(term <= 0)
        elif self.attributes['bounds']:
            bounds = self.bounds
            lower_bounds, upper_bounds = bounds
            # Create masks if -inf or inf is present in the bounds
            lower_bound_mask = (lower_bounds != -np.inf)
            upper_bound_mask = (upper_bounds != np.inf)

            if np.any(lower_bound_mask):
                # At least one valid lower bound,
                # so we apply the constraint only to those entries
                if self.ndim > 0:
                    constraints.append(term[lower_bound_mask] >= lower_bounds[lower_bound_mask])
                else:
                    constraints.append(term >= lower_bounds)
            if np.any(upper_bound_mask):
                # At least one valid upper bound,
                # so we apply the constraint only to those entries
                if self.ndim > 0:
                    constraints.append(term[upper_bound_mask] <= upper_bounds[upper_bound_mask])
                else:
                    constraints.append(term <= upper_bounds)

    @property
    def domain(self) -> list[Constraint]:
        """A list of constraints describing the closure of the region
           where the expression is finite.
        """
        # Default is full domain.
        domain = []
        # Add constraints from bounds.
        self._bound_domain(self, domain)
        # Add positive/negative semidefiniteness constraints.
        if self.attributes['PSD']:
            domain.append(self >> 0)
        elif self.attributes['NSD']:
            domain.append(self << 0)
        return domain

    def project(self, val, sparse_path=False):
        """Project value onto the attribute set of the leaf.

        A sensible idiom is ``leaf.value = leaf.project(val)``.

        Parameters
        ----------
        val : numeric type
            The value assigned.

        Returns
        -------
        numeric type
            The value rounded to the attribute type.
        """
        # Only one attribute can be active at once (besides real,
        # nonpos/nonneg, and bool/int).
        if not self.is_complex():
            val = np.real(val)

        if self.attributes['nonpos'] and self.attributes['nonneg']:
            return 0*val
        elif self.attributes['nonpos'] or self.attributes['neg']:
            return np.minimum(val, 0.)
        elif self.attributes['nonneg'] or self.attributes['pos']:
            return np.maximum(val, 0.)
        elif self.attributes['bounds']:
            return np.clip(val, self.bounds[0], self.bounds[1])
        elif self.attributes['imag']:
            return np.imag(val)*1j
        elif self.attributes['complex']:
            return val.astype(complex)
        elif self.attributes['boolean']:
            # TODO(akshayka): respect the boolean indices.
            return np.round(np.clip(val, 0., 1.))
        elif self.attributes['integer']:
            # TODO(akshayka): respect the integer indices.
            # also, a variable may be integer in some indices and
            # boolean in others.
            return np.round(val)
        elif self.attributes['diag']:
            if intf.is_sparse(val):
                val = val.diagonal()
            else:
                val = np.diag(val)
            return sp.diags([val], [0])
        elif self.attributes['hermitian']:
            return (val + np.conj(val).T)/2.
        elif any([self.attributes[key] for
                  key in ['symmetric', 'PSD', 'NSD']]):
            if val.dtype.kind in 'ib':
                val = val.astype(float)
            val = val + val.T
            val /= 2.
            if self.attributes['symmetric']:
                return val
            w, V = LA.eigh(val)
            if self.attributes['PSD']:
                bad = w < 0
                if not bad.any():
                    return val
                w[bad] = 0
            else:  # NSD
                bad = w > 0
                if not bad.any():
                    return val
                w[bad] = 0
            return (V * w).dot(V.T)
        elif self.attributes['sparsity'] and not sparse_path:
            warnings.warn('Accessing a sparse CVXPY expression via a dense representation.'
                          ' Please report this as a bug to the CVXPY Discord or GitHub.',
                          RuntimeWarning, 3)
            new_val = np.zeros(self.shape)
            new_val[self.sparse_idx] = val[self.sparse_idx]
            return new_val
        else:
            return val

    # Getter and setter for parameter value.
    def save_value(self, val, sparse_path=False) -> None:
        if self.sparse_idx is not None and not sparse_path:
            self._value = sp.coo_array((val[self.sparse_idx], self.sparse_idx), shape=self.shape)
        elif self.sparse_idx is not None and sparse_path:
            self._value = val.data
        else:
            self._value = val

    @property
    def value(self) -> Optional[np.ndarray]:
        """The numeric value of the expression."""
        if self.sparse_idx is None:
            return self._value
        else:
            warnings.warn('Reading from a sparse CVXPY expression via `.value` is discouraged.'
                          ' Use `.value_sparse` instead', RuntimeWarning, 1)
            if self._value is None:
                return None
            val = np.zeros(self.shape)
            val[self.sparse_idx] = self._value.data
            return val

    @value.setter
    def value(self, val) -> None:
        if self.sparse_idx is not None:
            warnings.warn('Writing to a sparse CVXPY expression via `.value` is discouraged.'
                          ' Use `.value_sparse` instead', RuntimeWarning, 1)
        self.save_value(self._validate_value(val))

    @property
    def value_sparse(self) -> Optional[...]:
        """The numeric value of the expression if it is a sparse variable."""
        if self._value is None:
            return None
        if isinstance(self._value, np.ndarray):
            return sp.coo_array((self._value, self.sparse_idx), shape=self.shape)
        else:
            return self._value

    @value_sparse.setter
    def value_sparse(self, val) -> None:
        if isinstance(val, sp.spmatrix):
            val = sp.coo_array(val)
        elif not isinstance(val, sp.coo_array):
            if isinstance(val, (np.ndarray)) \
                    and val.shape == (len(self.sparse_idx[0]),):
                raise ValueError(
                    'Invalid type for assigning value_sparse.'
                    'Try using `'
                    'expr.value_sparse = scipy.sparse.coo_array((values, expr)) instead.')
            raise ValueError(
                'Invalid type for assigning value_sparse.'
                f'Recieved: {type(val)} Expected scipy.sparse.coo_array.'
                f' Instantiate with scipy.sparse.coo_array((value_array, coordinates))'
                )
        self.save_value(self._validate_value(val, True), True)



    def project_and_assign(self, val) -> None:
        """Project and assign a value to the variable.
        """
        self.save_value(self.project(val))

    def _validate_value(self, val, sparse_path=False):
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
            # Convert val to ndarray or sparse matrix.
            val = intf.convert(val)
            if intf.shape(val) != self.shape:
                raise ValueError(
                    "Invalid dimensions %s for %s value." %
                    (intf.shape(val), self.__class__.__name__)
                )
            if sparse_path:
                val.sum_duplicates()
                coords_val = get_coords(val)
                if len(coords_val) != len(self.sparse_idx) or \
                     any((a != b).any() for a, b in zip(coords_val, self.sparse_idx)):
                    raise ValueError(
                        'Invalid sparsity pattern %s for %s value.' %
                        (get_coords(val), self.__class__.__name__)
                    )
            projection = self.project(val, sparse_path)
            # ^ might be a numpy array, or sparse scipy matrix.
            delta = np.abs(val - projection)
            # ^ might be a numpy array, scipy matrix, or sparse scipy matrix.
            if intf.is_sparse(delta):
                # ^ based on current implementation of project(...),
                #   is is not possible for this Leaf to be PSD/NSD *and*
                #   a sparse matrix.
                close_enough = np.allclose(delta.data, 0,
                                           atol=SPARSE_PROJECTION_TOL)
                # ^ only check for near-equality on nonzero values.
            else:
                # the data could be a scipy matrix, or a numpy array.
                # First we convert to a numpy array.
                delta = np.array(delta)
                # Now that we have the residual, we need to measure it
                # in some canonical way.
                if self.attributes['PSD'] or self.attributes['NSD']:
                    # For PSD/NSD Leafs, we use the largest-singular-value norm.
                    close_enough = LA.norm(delta, ord=2) <= PSD_NSD_PROJECTION_TOL
                else:
                    # For all other Leafs we use the infinity norm on
                    # the vectorized Leaf.
                    close_enough = np.allclose(delta, 0,
                                               atol=GENERAL_PROJECTION_TOL)
            if not close_enough:
                if self.attributes['nonneg']:
                    attr_str = 'nonnegative'
                elif self.attributes['pos']:
                    attr_str = 'positive'
                elif self.attributes['nonpos']:
                    attr_str = 'nonpositive'
                elif self.attributes['neg']:
                    attr_str = 'negative'
                elif self.attributes['sparsity']:
                    attr_str = 'zero outside of sparsity pattern'
                elif self.attributes['diag']:
                    attr_str = 'diagonal'
                elif self.attributes['PSD']:
                    attr_str = 'positive semidefinite'
                elif self.attributes['NSD']:
                    attr_str = 'negative semidefinite'
                elif self.attributes['imag']:
                    attr_str = 'imaginary'
                elif self.attributes['bounds']:
                    attr_str = 'in bounds'
                else:
                    attr_str = ([k for (k, v) in self.attributes.items() if v] + ['real'])[0]
                raise ValueError(
                    "%s value must be %s." % (self.__class__.__name__, attr_str)
                )
        return val

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?"""
        return self.attributes['PSD']

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?"""
        return self.attributes['NSD']

    def is_diag(self) -> bool:
        """Is the expression a diagonal matrix?"""
        return self.attributes['diag']

    def is_quadratic(self) -> bool:
        """Leaf nodes are always quadratic."""
        return True

    def has_quadratic_term(self) -> bool:
        """Leaf nodes are not quadratic terms."""
        return False

    def is_pwl(self) -> bool:
        """Leaf nodes are always piecewise linear."""
        return True

    def is_dpp(self, context: str = 'dcp') -> bool:
        """The expression is a disciplined parameterized expression.

           context: dcp or dgp
        """
        return True

    def atoms(self) -> list[Atom]:
        return []

    def _ensure_valid_bounds(self, value) -> Iterable | None:
        # In case for a constant or no bounds
        if value is None:
            return

        # Check that bounds is an iterable of two items
        if not isinstance(value, Iterable) or len(value) != 2:
            raise ValueError("Bounds should be a list of two items.")

        # Check that bounds contains two scalars or two arrays with matching shapes.
        for val in value:
            valid_array = isinstance(val, np.ndarray) and val.shape == self.shape
            if not (val is None or np.isscalar(val) or valid_array):
                raise ValueError(
                    "Bounds should be None, scalars, or arrays with the "
                    "same dimensions as the variable/parameter."
                )

        # Promote upper and lower bounds to arrays.
        none_bounds = [-np.inf, np.inf]
        for idx, val in enumerate(value):
            if val is None:
                value[idx] = np.full(self.shape, none_bounds[idx])
            elif np.isscalar(val):
                value[idx] = np.full(self.shape, val)

        # Upper bound cannot be -np.inf.
        if np.any(value[1] == -np.inf):
            raise ValueError("-np.inf is not feasible as an upper bound.")
        # Lower bound cannot be np.inf.
        if np.any(value[0] == np.inf):
            raise ValueError("np.inf is not feasible as a lower bound.")

        # Check that upper_bound >= lower_bound
        if np.any(value[0] > value[1]):
            raise ValueError("Invalid bounds: some upper bounds are less "
                             "than corresponding lower bounds.")

        if np.any(np.isnan(value[0])) or np.any(np.isnan(value[1])):
            raise ValueError("np.nan is not feasible as lower "
                                "or upper bound.")

        return value
