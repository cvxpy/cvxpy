"""
Copyright 2017 Robin Verschueren, 2017 Akshay Agrawal

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


from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.vec import vec as vec_atom
from cvxpy.cvxcore.python import canonInterface
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.reduction import Reduction


def extract_lower_bounds(variables: list, var_size: int) -> Optional[np.ndarray]:
    """Coalesces lower bounds for the variables.

    Parameters
    ----------
    variables: A list of the variables present in the problem.
    var_size: Size of the coalesced variable.
    """
    # No bounds case.
    bounds_present = any([var._has_lower_bounds() for var in variables])
    if not bounds_present:
        return None

    lower_bounds = np.full(var_size, -np.inf)
    vert_offset = 0
    for x in variables:
        if x.attributes["bounds"] is not None:
            # Store lower bound in Fortran order.
            # Use broadcast_to for memory-efficient scalar bounds.
            var_lower_bound = np.broadcast_to(x.attributes['bounds'][0], x.shape)
            flattened = np.reshape(var_lower_bound, x.size, order="F")
            lower_bounds[vert_offset:vert_offset+x.size] = flattened
        if x.is_nonneg():
            np.maximum(lower_bounds[vert_offset:vert_offset+x.size], 0,
                       out=lower_bounds[vert_offset:vert_offset+x.size])
        vert_offset += x.size
    return lower_bounds


def extract_upper_bounds(variables: list, var_size: int) -> Optional[np.ndarray]:
    """Coalesces upper bounds for the variables.

    Parameters
    ----------
    variables: A list of the variables present in the problem.
    var_size: Size of the coalesced variable.
    """
    # No bounds case.
    bounds_present = any([var._has_upper_bounds() for var in variables])
    if not bounds_present:
        return None

    upper_bounds = np.full(var_size, np.inf)
    vert_offset = 0
    for x in variables:
        if x.attributes["bounds"] is not None:
            # Store upper bound in Fortran order.
            # Use broadcast_to for memory-efficient scalar bounds.
            var_upper_bound = np.broadcast_to(x.attributes['bounds'][1], x.shape)
            flattened = np.reshape(var_upper_bound, x.size, order="F")
            upper_bounds[vert_offset:vert_offset+x.size] = flattened
        if x.is_nonpos():
            np.minimum(upper_bounds[vert_offset:vert_offset+x.size], 0,
                       out=upper_bounds[vert_offset:vert_offset+x.size])
        vert_offset += x.size
    return upper_bounds


def extract_mip_idx(variables) -> Tuple[List[int], List[int]]:
    """
    Coalesces bool, int indices for variables.
    The indexing scheme assumes that the variables will be coalesced into
    a single one-dimensional variable, with each variable being reshaped
    in Fortran order.
    """
    boolean_idx, integer_idx, offset = [], [], 0
    for x in variables:
        ravel_shape = max(x.shape, (1,))
        if x.boolean_idx:
            ravel_idx = np.ravel_multi_index(x.boolean_idx, ravel_shape, order='F')
            boolean_idx += [(idx + offset,) for idx in ravel_idx]
        if x.integer_idx:
            ravel_idx = np.ravel_multi_index(x.integer_idx, ravel_shape, order='F')
            integer_idx += [(idx + offset,) for idx in ravel_idx]
        offset += x.size
    return boolean_idx, integer_idx


def _has_parametric_bounds(variables) -> bool:
    """Check if any variable has an Expression (parametric) bound."""
    for var in variables:
        if var.bounds is not None:
            for b in var.bounds:
                if isinstance(b, Expression):
                    return True
    return False


def extract_bounds_tensor(
    variables: list,
    var_size: int,
    param_to_size: dict,
    param_id_map: dict,
    canon_backend: str | None,
    which: str,
) -> sp.sparray:
    """Build a sparse tensor mapping param_vec -> bounds_vec.

    For each variable's bound slot, we create a LinOp expression whose
    canonical form, when multiplied by the parameter vector, yields the
    bound value for that variable slice.

    Parameters
    ----------
    variables : list
        The problem's variables in order.
    var_size : int
        Total flattened variable size.
    param_to_size : dict
        Map from parameter id to parameter size.
    param_id_map : dict
        Map from parameter id to column offset in parameter vector.
    canon_backend : str or None
        Which canonicalization backend to use.
    which : str
        'lower' or 'upper'.

    Returns
    -------
    sp.sparray
        Sparse matrix of shape (var_size, param_size + 1). The last column
        holds the constant (parameter-free) part of the bounds.
    """
    assert which in ('lower', 'upper')
    bound_idx = 0 if which == 'lower' else 1
    default_val = -np.inf if which == 'lower' else np.inf

    op_list = []
    for x in variables:
        bound_expr = None
        if which == 'lower' and x.is_nonneg():
            bound_expr = Constant(np.zeros(x.size))
        elif which == 'upper' and x.is_nonpos():
            bound_expr = Constant(np.zeros(x.size))
        elif x.attributes["bounds"] is not None:
            b = x.attributes['bounds'][bound_idx]
            if isinstance(b, Expression):
                # Flatten in Fortran order to match variable layout.
                if b.is_scalar() and x.size > 1:
                    # Broadcast scalar expression to variable size.
                    b = Promote(b, x.shape)
                if b.ndim > 1:
                    bound_expr = vec_atom(b, order='F')
                else:
                    bound_expr = b
            else:
                # Use broadcast_to for memory-efficient scalar bounds.
                b_broadcast = np.broadcast_to(b, x.shape)
                flattened = np.reshape(b_broadcast, x.size, order="F")
                bound_expr = Constant(flattened)
        else:
            bound_expr = Constant(np.full(x.size, default_val))

        op_list.append(bound_expr.canonical_form[0])

    # Build the problem matrix with x_length=0 (no variable columns,
    # only the constant/parameter mapping).
    tensor = canonInterface.get_problem_matrix(
        op_list,
        var_length=0,
        id_to_col={},
        param_to_size=param_to_size,
        param_to_col=param_id_map,
        constr_length=var_size,
        canon_backend=canon_backend,
    )
    return tensor


class MatrixStuffing(Reduction):
    """Stuffs a problem into a standard form for a family of solvers."""

    def apply(self, problem) -> None:
        """Returns a stuffed problem.

        The returned problem is a minimization problem in which every
        constraint in the problem has affine arguments that are expressed in
        the form A @ x + b.


        Parameters
        ----------
        problem: The problem to stuff; the arguments of every constraint
            must be affine

        Returns
        -------
        Problem
            The stuffed problem
        InverseData
            Data for solution retrieval
        """
    def invert(self, solution, inverse_data):
        raise NotImplementedError()

    def stuffed_objective(self, problem, inverse_data):
        raise NotImplementedError()
