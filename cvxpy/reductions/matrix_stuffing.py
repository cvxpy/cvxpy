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
        if x.is_nonneg():
            lower_bounds[vert_offset:vert_offset+x.size] = 0
        elif x.attributes["bounds"] is not None:
            # Store lower bound in Fortran order.
            var_lower_bound = x.attributes['bounds'][0]
            flattened = np.reshape(var_lower_bound, x.size, order="F")
            lower_bounds[vert_offset:vert_offset+x.size] = flattened
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
        if x.is_nonpos():
            upper_bounds[vert_offset:vert_offset+x.size] = 0
        elif x.attributes["bounds"] is not None:
            # Store upper bound in Fortran order.
            var_upper_bound = x.attributes['bounds'][1]
            flattened = np.reshape(var_upper_bound, x.size, order="F")
            upper_bounds[vert_offset:vert_offset+x.size] = flattened
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
