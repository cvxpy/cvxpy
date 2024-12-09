"""
Copyright 2018 Akshay Agrawal

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

from collections import defaultdict
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.cvxcore.python import canonInterface


def lower_ineq_to_nonpos(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonPos(lhs - rhs, constr_id=inequality.constr_id)


def lower_ineq_to_nonneg(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonNeg(rhs - lhs, constr_id=inequality.constr_id)


def lower_equality(equality):
    lhs = equality.args[0]
    rhs = equality.args[1]
    return Zero(lhs - rhs, constr_id=equality.constr_id)


def nonpos2nonneg(nonpos):
    return NonNeg(-nonpos.expr, constr_id=nonpos.constr_id)


def special_index_canon(expr, args):
    select_mat = expr._select_mat
    final_shape = expr._select_mat.shape
    select_vec = np.reshape(select_mat, select_mat.size, order='F')
    # Select the chosen entries from expr.
    arg = args[0]
    identity = sp.eye(arg.size).tocsc()
    lowered = reshape(identity[select_vec] @ vec(arg, order='F'), final_shape, order='F')
    return lowered, []


def are_args_affine(constraints) -> bool:
    return all(arg.is_affine() for constr in constraints
               for arg in constr.args)


def group_constraints(constraints):
    """Organize the constraints into a dictionary keyed by constraint names.

    Parameters
    ---------
    constraints : list of constraints

    Returns
    -------
    dict
        A dict keyed by constraint types where dict[cone_type] maps to a list
        of exactly those constraints that are of type cone_type.
    """
    constr_map = defaultdict(list)
    for c in constraints:
        constr_map[type(c)].append(c)
    return constr_map


class ReducedMat:
    """Utility class for condensing the mapping from parameters to problem data.

    For maximum efficiency of representation and application, the mapping from
    parameters to problem data must be condensed. It begins as a CSC sparse matrix
    matrix_data, such that multiplying by a parameter vector gives the problem data.
    The row index array and column pointer array are saved as problem_data_index,
    and a CSR matrix reduced_mat that when multiplied by a parameter vector gives
    the values array. The ReducedMat class caches the condensed representation
    and provides a method for multiplying by a parameter vector.

    This class consolidates code from ParamConeProg and ParamQuadProg.

    Attributes
    ----------
    matrix_data : SciPy CSC sparse matrix
       A matrix representing the mapping from parameter to problem data.
    var_len : int
       The length of the problem variable.
    quad_form: (optional) if True, consider quadratic form matrix P
    """

    def __init__(self, matrix_data, var_len: int, quad_form: bool = False) -> None:
        self.matrix_data = matrix_data
        self.var_len = var_len
        self.quad_form = quad_form
        # A CSR sparse matrix with redundant rows removed
        # such that reduced_mat @ param = problem_data values array.
        self.reduced_mat = None
        # A tuple containing the following:
        # CSC indices for the problem data matrix
        # CSC indptr for the problem data matrix
        self.problem_data_index = None
        # The rows in the map from parameters to problem data that
        # have any nonzeros.
        self.mapping_nonzero = None

    def cache(self, keep_zeros: bool = False) -> None:
        """Cache computed attributes if not present.

        Parameters
        ----------
            keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected.
        """
        # Short circuit null case.
        if self.matrix_data is None:
            return

        if self.reduced_mat is None:
            # Form a reduced representation of the mapping,
            # for faster application of parameters.
            if np.prod(self.matrix_data.shape) != 0:
                reduced_mat, indices, indptr, shape = (
                    canonInterface.reduce_problem_data_tensor(
                        self.matrix_data, self.var_len, self.quad_form))
                self.reduced_mat = reduced_mat
                self.problem_data_index = (indices, indptr, shape)
            else:
                self.reduced_mat = self.matrix_data
                self.problem_data_index = None

        if keep_zeros and self.mapping_nonzero is None:
            self.mapping_nonzero = canonInterface.A_mapping_nonzero_rows(
                self.matrix_data, self.var_len)

    def get_matrix_from_tensor(self, param_vec: np.ndarray, with_offset: bool = True) -> Tuple:
        """Wraps get_matrix_from_tensor in canonInterface.

        Parameters
        ----------
            param_vec: flattened parameter vector
            with_offset: (optional) return offset. Defaults to True.

        Returns
        -------
            A tuple (A, b), where A is a matrix with `var_length` columns
            and b is a flattened NumPy array representing the constant offset.
            If with_offset=False, returned b is None.
        """
        return canonInterface.get_matrix_from_tensor(
            self.reduced_mat, param_vec, self.var_len,
            nonzero_rows=self.mapping_nonzero,
            with_offset=with_offset,
            problem_data_index=self.problem_data_index)
