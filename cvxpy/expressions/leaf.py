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
from cvxpy.utilities.bounds import coords_equal
from cvxpy.utilities.coo_array_compat import get_coords
from cvxpy.utilities.warn import warn


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
            if not isinstance(d, numbers.Integral) or d < 0:
                raise ValueError("Invalid dimensions %s." % (shape,))
        shape = tuple(shape)
        self._shape = shape
        super(Leaf, self).__init__()

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
            self._sparse_high_fill_in = (len(self.sparse_idx[0]) / np.prod(self.shape) <= 0.25)
        else:
            self.sparse_idx = None
        # count number of attributes
        self.num_attributes = sum(1 for k, v in self.attributes.items() if v)
        dim_reducing_attr = ['diag', 'symmetric', 'PSD', 'NSD', 'hermitian', 'sparsity']
        if sum(1 for k in dim_reducing_attr if self.attributes[k]) > 1:
            raise ValueError(
                "A CVXPY Variable cannot have more than one of the following attributes: "
                f"{dim_reducing_attr}"
            )
        sign_attrs = [k for k in ['pos', 'neg'] if self.attributes[k]]
        sparse_attrs = [k for k in ['sparsity', 'diag'] if self.attributes[k]]
        if sign_attrs and sparse_attrs:
            raise ValueError(
                f"Cannot combine {sign_attrs} with {sparse_attrs}. "
                "Sparsity and diag attributes force zeros, which contradicts "
                "strict positivity/negativity."
            )
        self._leaf_of_provenance = None
        self.args = []
        self.bounds = self._ensure_valid_bounds(bounds)
        self.attributes['bounds'] = self.bounds
        if value is not None:
            self.value = value

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
            if val is not False and val is not None:
                if isinstance(val, bool):
                    attr_str += ", %s=%s" % (attr, val)
                elif attr == 'bounds' and val is not None:
                    parts = []
                    for b in val:
                        if isinstance(b, expression.Expression):
                            parts.append(str(b))
                        else:
                            parts.append(np.array2string(
                                b,
                                edgeitems=s.PRINT_EDGEITEMS,
                                threshold=s.PRINT_THRESHOLD,
                                formatter={'float': lambda x: f'{x:.2f}'}))
                    attr_str += ", %s=(%s, %s)" % (attr, parts[0], parts[1])
                elif attr in ('sparsity', 'boolean', 'integer') and isinstance(val, Iterable):
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
            if isinstance(lower_bound, expression.Expression):
                return True
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
            if isinstance(upper_bound, expression.Expression):
                return True
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
        constraints: An existing list of constraints to append to.
        """
        if self.attributes['nonneg'] or self.attributes['pos']:
            constraints.append(term >= 0)
        if self.attributes['nonpos'] or self.attributes['neg']:
            constraints.append(term <= 0)
        if self.attributes['bounds']:
            bounds = self.bounds
            lower_bounds, upper_bounds = bounds

            # Helper to check if bounds are scalar (0-d array or Python scalar)
            def is_scalar_bound(b):
                return np.isscalar(b) or (hasattr(b, 'ndim') and b.ndim == 0)

            # Expression bounds (e.g. cp.Parameter): no masking needed
            if isinstance(lower_bounds, expression.Expression):
                constraints.append(term >= lower_bounds)
            elif sp.issparse(lower_bounds):
                # Sparse lower bounds: use the sparse indices directly.
                # Defensive COO conversion: bounds are stored as COO from
                # _ensure_valid_bounds, but re-canonicalize to be safe.
                sparse_lb = sp.coo_array(lower_bounds)
                sparse_lb.sum_duplicates()
                mask = sparse_lb.data != -np.inf
                if np.any(mask):
                    # Get coordinates for the finite bounds
                    indices = tuple(coord[mask] for coord in get_coords(sparse_lb))
                    constraints.append(term[indices] >= sparse_lb.data[mask])
            elif np.any(lower_bounds != -np.inf):
                # Scalar/0-d bounds apply to all elements uniformly
                if is_scalar_bound(lower_bounds):
                    constraints.append(term >= float(lower_bounds))
                elif self.ndim > 0:
                    lower_bound_mask = (lower_bounds != -np.inf)
                    constraints.append(term[lower_bound_mask] >= lower_bounds[lower_bound_mask])
                else:
                    constraints.append(term >= lower_bounds)

            if isinstance(upper_bounds, expression.Expression):
                constraints.append(term <= upper_bounds)
            elif sp.issparse(upper_bounds):
                # Sparse upper bounds: use the sparse indices directly.
                # Defensive COO conversion: see comment above for lower bounds.
                sparse_ub = sp.coo_array(upper_bounds)
                sparse_ub.sum_duplicates()
                mask = sparse_ub.data != np.inf
                if np.any(mask):
                    # Get coordinates for the finite bounds
                    indices = tuple(coord[mask] for coord in get_coords(sparse_ub))
                    constraints.append(term[indices] <= sparse_ub.data[mask])
            elif np.any(upper_bounds != np.inf):
                # Scalar/0-d bounds apply to all elements uniformly
                if is_scalar_bound(upper_bounds):
                    constraints.append(term <= float(upper_bounds))
                elif self.ndim > 0:
                    upper_bound_mask = (upper_bounds != np.inf)
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
        if not self.is_complex():
            val = np.real(val)
        # Skip the projection operation for more than one attribute
        if self.num_attributes > 1:
            return val
        if self.attributes['nonpos'] and self.attributes['nonneg']:
            return 0*val
        elif self.attributes['nonpos'] or self.attributes['neg']:
            return np.minimum(val, 0.)
        elif self.attributes['nonneg'] or self.attributes['pos']:
            return np.maximum(val, 0.)
        elif self.attributes['bounds']:
            if any(isinstance(b, expression.Expression) for b in self.bounds):
                # Cannot project with expression bounds; return as-is.
                return val
            return np.clip(val, self.bounds[0], self.bounds[1])
        elif self.attributes['imag']:
            return np.imag(val)*1j
        elif self.attributes['complex']:
            return val.astype(complex)
        elif self.attributes['boolean']:
            if hasattr(self, "boolean_idx"):
                new_val = np.atleast_1d(val.astype(np.float64, copy=True))
                new_val[self.boolean_idx] = np.round(np.clip(new_val[self.boolean_idx], 0., 1.))
                return new_val.reshape(val.shape) if val.ndim == 0 else new_val
        elif self.attributes['integer']:
            if hasattr(self, "integer_idx"):
                new_val = np.atleast_1d(val.astype(np.float64, copy=True))
                new_val[self.integer_idx] = np.round(new_val[self.integer_idx])
                return new_val.reshape(val.shape) if val.ndim == 0 else new_val
        elif self.attributes['diag']:
            if intf.is_sparse(val):
                val = val.diagonal()
            else:
                val = np.diag(val)
            return sp.diags_array([val], offsets=[0])
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
            warn('Accessing a sparse CVXPY expression via a dense representation.'
                  ' Please report this as a bug to the CVXPY Discord or GitHub.',
                  RuntimeWarning)
            new_val = np.zeros(self.shape)
            new_val[self.sparse_idx] = val[self.sparse_idx]
            return new_val
        else:
            return val

    # Getter and setter for parameter value.
    def save_value(self, val, sparse_path=False) -> None:
        if val is None:
            self._value = None
        elif self.sparse_idx is not None and not sparse_path:
            self._value = val[self.sparse_idx]
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
            warn('Reading from a sparse CVXPY expression via `.value` is discouraged.'
                  ' Use `.value_sparse` instead', RuntimeWarning)
            if self._value is None:
                return None
            val = np.zeros(self.shape, dtype=self._value.dtype)
            val[self.sparse_idx] = self._value
            return val

    @value.setter
    def value(self, val) -> None:
        if self.sparse_idx is not None and self._sparse_high_fill_in:
            warn('Writing to a sparse CVXPY expression via `.value` is discouraged.'
                  ' Use `.value_sparse` instead', RuntimeWarning)
        self.save_value(self._validate_value(val))

    @property
    def value_sparse(self) -> Optional[...]:
        """The numeric value of the expression if it is a sparse variable."""
        if self._value is None:
            return None
        return sp.coo_array((self._value, self.sparse_idx), shape=self.shape)

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
            with np.errstate(invalid='ignore'):
                delta = np.abs(val - projection)
            # ^ might be a numpy array, scipy matrix, or sparse scipy matrix.
            if intf.is_sparse(delta):
                # ^ based on current implementation of project(...),
                #   it is not possible for this Leaf to be PSD/NSD *and*
                #   a sparse matrix.
                # Handle inf - inf = NaN: replace NaN with 0 where val
                # and projection agree (both +inf or both -inf).
                if delta.data.size > 0:
                    nan_mask = np.isnan(delta.data)
                    if np.any(nan_mask):
                        val_sp = val.tocsr() if hasattr(val, 'tocsr') else val
                        proj_sp = projection.tocsr() if hasattr(projection, 'tocsr') else projection
                        val_arr = np.asarray(val_sp[delta.nonzero()]).ravel()
                        proj_arr = np.asarray(proj_sp[delta.nonzero()]).ravel()
                        delta.data[nan_mask & (val_arr == proj_arr)] = 0.0
                close_enough = np.allclose(delta.data, 0,
                                           atol=SPARSE_PROJECTION_TOL)
                # ^ only check for near-equality on nonzero values.
            else:
                # the data could be a scipy matrix, or a numpy array.
                # First we convert to a numpy array.
                delta = np.array(delta)
                # Handle inf - inf = NaN: replace NaN with 0 where val
                # and projection agree (both +inf or both -inf).
                nan_mask = np.isnan(delta)
                if np.any(nan_mask):
                    delta = np.where(
                        nan_mask & (np.asarray(val) == np.asarray(projection)),
                        0.0, delta
                    )
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

    def attributes_were_lowered(self) -> bool:
        """True iff this leaf was generated when lowering a leaf with attributes."""
        return self._leaf_of_provenance is not None

    def set_leaf_of_provenance(self, leaf: Leaf) -> None:
        self._leaf_of_provenance = leaf

    def leaf_of_provenance(self) -> Leaf | None:
        """Returns a leaf with attributes from which this leaf was generated."""
        return self._leaf_of_provenance

    @property
    def _has_dim_reducing_attr(self) -> bool:
        return (self.sparse_idx is not None or self.attributes['diag'] or
                self.attributes['symmetric'] or self.attributes['PSD'] or
                self.attributes['NSD'])

    @property
    def _reduced_size(self) -> int:
        if self.sparse_idx is not None:
            return len(self.sparse_idx[0])
        elif self.attributes['diag']:
            return self.shape[0]
        elif self.attributes['symmetric'] or self.attributes['PSD'] or self.attributes['NSD']:
            return self.shape[0] * (self.shape[0] + 1) // 2
        return self.size

    def _validate_sparse_bound(self, val):
        """Validate a single sparse bound entry.

        Checks that the sparse bound has matching shape and sparsity pattern.

        Raises
        ------
        ValueError
            If the sparse bound is invalid.
        """
        if val.shape != self.shape:
            raise ValueError(
                "Sparse bounds must have the same shape as the variable."
            )
        if self.sparse_idx is not None:
            coo = sp.coo_array(val)
            coo.sum_duplicates()
            val_coords = get_coords(coo)
            if not coords_equal(val_coords, self.sparse_idx):
                raise ValueError(
                    "Sparse bounds must have the same sparsity pattern "
                    "as the sparse variable."
                )
        else:
            raise ValueError(
                "Sparse bounds are only supported for sparse variables."
            )

    @staticmethod
    def _promote_bounds(value):
        """Promote bound entries to canonical form.

        None → 0-d array (-inf/inf), scalars → 0-d arrays,
        sparse → COO arrays, dense arrays kept as-is.

        Returns
        -------
        list
            Two-element list of promoted bounds [lb, ub].
        """
        none_defaults = [-np.inf, np.inf]
        promoted = []
        for idx, val in enumerate(value):
            if val is None:
                promoted.append(np.array(none_defaults[idx]))
            elif sp.issparse(val):
                promoted.append(sp.coo_array(val))
            elif np.isscalar(val):
                promoted.append(np.array(val))
            else:
                promoted.append(val)
        return promoted

    @staticmethod
    def _check_bound_feasibility(lb, ub, has_structural_zeros):
        """Check that promoted bounds are feasible.

        Validates: no -inf upper bounds, no +inf lower bounds,
        lb <= ub, no NaN, and structural zero consistency.

        Raises
        ------
        ValueError
            If bounds are infeasible.
        """
        lb_data = lb.data if sp.issparse(lb) else lb
        ub_data = ub.data if sp.issparse(ub) else ub

        if np.any(ub_data == -np.inf):
            raise ValueError("-np.inf is not feasible as an upper bound.")
        if np.any(lb_data == np.inf):
            raise ValueError("np.inf is not feasible as a lower bound.")

        # Check that upper_bound >= lower_bound.
        # For mixed sparse/scalar bounds, we only check on-pattern entries here.
        # Off-pattern entries are implicitly 0, and the scalar bounds validation
        # below ensures scalars contain 0 (lb <= 0, ub >= 0), so off-pattern
        # entries are always feasible.
        if sp.issparse(lb) and sp.issparse(ub):
            if np.any(lb.data > ub.data):
                raise ValueError("Invalid bounds: some upper bounds are less "
                                 "than corresponding lower bounds.")
        elif np.any(lb_data > ub_data):
            raise ValueError("Invalid bounds: some upper bounds are less "
                             "than corresponding lower bounds.")

        if np.any(np.isnan(lb_data)) or np.any(np.isnan(ub_data)):
            raise ValueError("np.nan is not feasible as lower "
                             "or upper bound.")

        # For variables with structural zeros and scalar bounds, require
        # that 0 is between lb and ub. The structurally zero entries are
        # fixed at 0, so bounds that exclude 0 would be inconsistent.
        if has_structural_zeros:
            lb_is_scalar = isinstance(lb, np.ndarray) and lb.ndim == 0
            ub_is_scalar = isinstance(ub, np.ndarray) and ub.ndim == 0
            if lb_is_scalar and float(lb) > 0:
                raise ValueError(
                    "Scalar lower bound for a sparse or diagonal variable "
                    "must be <= 0, since the structurally zero entries "
                    "are fixed at 0."
                )
            if ub_is_scalar and float(ub) < 0:
                raise ValueError(
                    "Scalar upper bound for a sparse or diagonal variable "
                    "must be >= 0, since the structurally zero entries "
                    "are fixed at 0."
                )

    def _ensure_valid_bounds(self, value) -> Iterable | None:
        if value is None:
            return

        if not isinstance(value, Iterable) or len(value) != 2:
            raise ValueError("Bounds should be a list of two items.")

        value = list(value)

        has_expr_bound = any(
            isinstance(val, expression.Expression) for val in value
        )

        # Variables with structural zeros: off-pattern entries (sparse)
        # or off-diagonal entries (diag) are fixed at 0.
        has_structural_zeros = (
            self.attributes.get('sparsity') or self.attributes.get('diag')
        )

        if has_expr_bound:
            if has_structural_zeros:
                raise ValueError(
                    "Expression bounds are not yet supported for sparse "
                    "or diagonal variables. If you need this feature, "
                    "please contact the CVXPY developers at "
                    "https://github.com/cvxpy/cvxpy/issues as we have "
                    "design questions we want user feedback on."
                )
            # Validate Expression bounds: must be scalar or matching shape,
            # and must not depend on any Variable.
            for idx, val in enumerate(value):
                if isinstance(val, expression.Expression):
                    if val.variables():
                        raise ValueError(
                            "Parametric bounds must not depend on Variables. "
                            "Use Parameters or numeric values instead."
                        )
                    if not (val.is_scalar() or val.shape == self.shape):
                        raise ValueError(
                            "Expression bounds must be scalar or have the "
                            "same dimensions as the variable."
                        )
                elif val is None:
                    none_bounds = [-np.inf, np.inf]
                    value[idx] = np.array(none_bounds[idx])
                elif np.isscalar(val):
                    value[idx] = np.array(val)
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    pass
                else:
                    valid_array = isinstance(val, np.ndarray) and val.shape == self.shape
                    if not valid_array:
                        raise ValueError(
                            "Bounds should be None, scalars, arrays, or "
                            "CVXPY Expressions with matching dimensions."
                        )
            return value

        # --- Non-expression (numeric) bounds path ---
        # Convert list-like bounds to numpy arrays for validation (skip sparse)
        for idx, val in enumerate(value):
            if val is not None and not np.isscalar(val) and not sp.issparse(val):
                if not isinstance(val, np.ndarray):
                    try:
                        value[idx] = np.asarray(val)
                    except (TypeError, ValueError):
                        pass  # Will fail validation below

        def is_scalar_like(v):
            return np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0)

        # Validate shapes and sparsity patterns.
        for idx, val in enumerate(value):
            if sp.issparse(val):
                self._validate_sparse_bound(val)
            else:
                valid_scalar = is_scalar_like(val)
                valid_array = isinstance(val, np.ndarray) and val.shape == self.shape
                if not (val is None or valid_scalar or valid_array):
                    raise ValueError(
                        "Bounds should be None, scalars, or arrays with the "
                        "same dimensions as the variable/parameter."
                    )
                if valid_array and has_structural_zeros:
                    raise ValueError(
                        "Dense array bounds are not supported for sparse "
                        "or diagonal variables. Use scalar bounds instead."
                    )

        promoted = self._promote_bounds(value)
        self._check_bound_feasibility(promoted[0], promoted[1], has_structural_zeros)
        return promoted

    def get_bounds(self) -> tuple[np.ndarray | sp.sparray, np.ndarray | sp.sparray]:
        """Return bounds (lower, upper) for this leaf.

        For Variables: combines explicit bounds with sign attributes.
        For Constants: returns (value, value).
        For Parameters: combines explicit bounds with sign attributes.

        This method is memory-efficient: it uses broadcast views for uniform
        scalar bounds and preserves sparse matrices without densifying.

        Returns
        -------
        tuple of (np.ndarray | sp.sparray)
            (lower_bound, upper_bound) arrays broadcastable to self.shape.
            For sparse variables with sparse bounds, returns sparse arrays.
        """
        # Determine effective lower and upper bounds, starting with unbounded.
        # We track scalar vs array bounds to enable memory-efficient broadcast.
        lb_val: float = -np.inf
        ub_val: float = np.inf
        lb_arr = None  # Non-None if bounds are non-uniform (array or sparse)
        ub_arr = None

        # Apply bounds attribute if present (skip Expression bounds,
        # which are symbolic and enforced at solve time).
        bounds_attr = self.attributes['bounds']
        if bounds_attr is not None:
            bound_lb, bound_ub = bounds_attr[0], bounds_attr[1]
            if not isinstance(bound_lb, expression.Expression):
                # Check for scalar or 0-d array (memory-efficient bounds)
                if np.isscalar(bound_lb) or (hasattr(bound_lb, 'ndim') and bound_lb.ndim == 0):
                    lb_val = max(lb_val, float(bound_lb))
                elif sp.issparse(bound_lb):
                    # Sparse lower bound - use sparse-aware max
                    if lb_val > -np.inf:
                        lb_arr = bound_lb.maximum(lb_val)
                    else:
                        lb_arr = bound_lb
                else:
                    # Array bound
                    lb_arr = np.maximum(lb_val, bound_lb)
            if not isinstance(bound_ub, expression.Expression):
                # Check for scalar or 0-d array (memory-efficient bounds)
                if np.isscalar(bound_ub) or (hasattr(bound_ub, 'ndim') and bound_ub.ndim == 0):
                    ub_val = min(ub_val, float(bound_ub))
                elif sp.issparse(bound_ub):
                    # Sparse upper bound - use sparse-aware min
                    if ub_val < np.inf:
                        ub_arr = bound_ub.minimum(ub_val)
                    else:
                        ub_arr = bound_ub
                else:
                    # Array bound
                    ub_arr = np.minimum(ub_val, bound_ub)

        # Apply sign attributes
        if self.attributes['nonneg'] or self.attributes['pos']:
            if lb_arr is not None:
                if sp.issparse(lb_arr):
                    lb_arr = lb_arr.maximum(0)
                else:
                    lb_arr = np.maximum(lb_arr, 0)
            else:
                lb_val = max(lb_val, 0)
        if self.attributes['nonpos'] or self.attributes['neg']:
            if ub_arr is not None:
                if sp.issparse(ub_arr):
                    ub_arr = ub_arr.minimum(0)
                else:
                    ub_arr = np.minimum(ub_arr, 0)
            else:
                ub_val = min(ub_val, 0)

        # For boolean variables, bounds are [0, 1]
        if self.attributes['boolean'] is True:
            if lb_arr is not None:
                if sp.issparse(lb_arr):
                    lb_arr = lb_arr.maximum(0)
                else:
                    lb_arr = np.maximum(lb_arr, 0)
            else:
                lb_val = max(lb_val, 0)
            if ub_arr is not None:
                if sp.issparse(ub_arr):
                    ub_arr = ub_arr.minimum(1)
                else:
                    ub_arr = np.minimum(ub_arr, 1)
            else:
                ub_val = min(ub_val, 1)

        # Build final bounds: use broadcast views for uniform scalars
        if lb_arr is not None:
            lb = lb_arr
        else:
            # Use memory-efficient broadcast view
            lb = np.broadcast_to(np.array(lb_val), self.shape)

        if ub_arr is not None:
            ub = ub_arr
        else:
            # Use memory-efficient broadcast view
            ub = np.broadcast_to(np.array(ub_val), self.shape)

        return (lb, ub)
