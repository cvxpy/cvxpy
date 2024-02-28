"""
Copyright 2017 Steven Diamond

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

import os

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend


def get_parameter_vector(param_size,
                         param_id_to_col,
                         param_id_to_size,
                         param_id_to_value_fn,
                         zero_offset: bool = False):
    """Returns a flattened parameter vector

    The flattened vector includes a constant offset (i.e, a 1).

    Parameters
    ----------
        param_size: The number of parameters
        param_id_to_col: A dict from parameter id to column offset
        param_id_to_size: A dict from parameter id to parameter size
        param_id_to_value_fn: A callable that returns a value for a parameter id
        zero_offset: (optional) if True, zero out the constant offset in the
                     parameter vector

    Returns
    -------
        A flattened NumPy array of parameter values, of length param_size + 1
    """
    #TODO handle parameters with structure.
    if param_size == 0:
        return None
    param_vec = np.zeros(param_size + 1)
    for param_id, col in param_id_to_col.items():
        if param_id == lo.CONSTANT_ID:
            if not zero_offset:
                param_vec[col] = 1
        else:
            value = param_id_to_value_fn(param_id).flatten(order='F')
            size = param_id_to_size[param_id]
            param_vec[col:col + size] = value
    return param_vec


def reduce_problem_data_tensor(A, var_length, quad_form: bool = False):
    """Reduce a problem data tensor, for efficient construction of the problem data

    If quad_form=False, the problem data tensor A is a matrix of shape (m, p), where p is the
    length of the parameter vector. The product A@param_vec gives the
    entries of the problem data matrix for a solver;
    the solver's problem data matrix has dimensions(n_constr, n_var + 1),
    and n_constr*(n_var + 1) = m. In other words, each row in A corresponds
    to an entry in the solver's data matrix.

    If quad_form=True, the problem data tensor A is a matrix of shape (m, p), where p is the
    length of the parameter vector. The product A@param_vec gives the
    entries of the quadratic form matrix P for a QP solver;
    the solver's quadratic matrix P has dimensions(n_var, n_var),
    and n_var*n_var = m. In other words, each row in A corresponds
    to an entry in the solver's quadratic form matrix P.

    This function removes the rows in A that are identically zero, since these
    rows correspond to zeros in the problem data. It also returns the indices
    and indptr to construct the problem data matrix from the reduced
    representation of A, and the shape of the problem data matrix.

    Let reduced_A be the sparse matrix returned by this function. Then the
    problem data can be computed using

       data : = reduced_A @ param_vec

    and the problem data matrix can be constructed with

        problem_data : = sp.csc_matrix(
            (data, indices, indptr), shape = shape)

    Parameters
    ----------
        A : A sparse matrix, the problem data tensor; must not have a 0 in its
            shape
        var_length: number of variables in the problem
        quad_form: (optional) if True, consider quadratic form matrix P

    Returns
    -------
        reduced_A: A CSR sparse matrix with redundant rows removed
        indices: CSC indices for the problem data matrix
        indptr: CSC indptr for the problem data matrix
        shape: the shape of the problem data matrix
    """
    # construct a reduced COO matrix
    A.eliminate_zeros()
    A_coo = A.tocoo()

    unique_old_row, reduced_row = np.unique(A_coo.row, return_inverse=True)
    reduced_A_shape = (unique_old_row.size, A_coo.shape[1])

    # remap the rows
    reduced_A = sp.coo_matrix(
        (A_coo.data, (reduced_row, A_coo.col)), shape=reduced_A_shape)

    # convert reduced_A to csr
    reduced_A = reduced_A.tocsr()

    nonzero_rows = unique_old_row
    n_cols = var_length
    # add one more column for the offset if not quad_form
    if not quad_form:
        n_cols += 1
    n_constr = A.shape[0] // (n_cols)
    shape = (n_constr, n_cols)
    indices = nonzero_rows % (n_constr)

    # cols holds the column corresponding to each row in nonzero_rows
    cols = nonzero_rows // n_constr

    # construction of the indptr: scan through cols, and find
    # the structure of the column index pointer
    indptr = np.zeros(n_cols + 1, dtype=np.int32)
    positions, counts = np.unique(cols, return_counts=True)
    indptr[positions+1] = counts
    indptr = np.cumsum(indptr)

    return reduced_A, indices, indptr, shape


def nonzero_csc_matrix(A):
    # this function returns (rows, cols) corresponding to nonzero entries in
    # A; an entry that is explicitly set to zero is treated as nonzero
    assert not np.isnan(A.data).any()

    # scipy drops rows, cols with explicit zeros; use nan as a sentinel
    # to prevent them from being dropped
    zero_indices = (A.data == 0)
    A.data[zero_indices] = np.nan

    # A.nonzero() returns (rows, cols) sorted in C-style order,
    # but (when A is a csc matrix) A.data is stored in Fortran-order, hence
    # the sorting below
    A_rows, A_cols = A.nonzero()
    ind = np.argsort(A_cols, kind='mergesort')
    A_rows = A_rows[ind]
    A_cols = A_cols[ind]

    A.data[zero_indices] = 0
    return A_rows, A_cols


def A_mapping_nonzero_rows(problem_data_tensor, var_length):
    # get the rows in the map from parameters to problem data that
    # have any nonzeros
    problem_data_tensor_csc = problem_data_tensor.tocsc()
    A_nrows = problem_data_tensor.shape[0] // (var_length + 1)
    A_ncols = var_length
    A_mapping = problem_data_tensor_csc[:A_nrows*A_ncols, :-1]
    # don't call nonzero_csc_matrix, because here we don't want to
    # count explicit zeros
    A_mapping_nonzero_rows, _ = A_mapping.nonzero()
    return np.unique(A_mapping_nonzero_rows)


def get_matrix_from_tensor(problem_data_tensor, param_vec,
                           var_length, nonzero_rows=None,
                           with_offset=True,
                           problem_data_index=None):
    """Applies problem_data_tensor to param_vec to obtain matrix and (optionally)
    the offset.

    This function applies problem_data_tensor to param_vec to obtain
    a matrix representation of the corresponding affine map.

    Parameters
    ----------
        problem_data_tensor: tensor returned from get_problem_matrix,
            representing a parameterized affine map
        param_vec: flattened parameter vector
        var_length: the number of variables
        nonzero_rows: (optional) rows in the part of problem_data_tensor
            corresponding to A that have nonzeros in them (i.e., rows that
            are affected by parameters); if not None, then the corresponding
            entries in A will have explicit zeros.
        with_offset: (optional) return offset. Defaults to True.
        problem_data_index: (optional) a tuple (indices, indptr, shape) for
            construction of the CSC matrix holding the problem data and offset

    Returns
    -------
        A tuple (A, b), where A is a matrix with `var_length` columns
        and b is a flattened NumPy array representing the constant offset.
        If with_offset=False, returned b is None.
    """
    if param_vec is None:
        flat_problem_data = problem_data_tensor
        if problem_data_index is not None:
            flat_problem_data = flat_problem_data.toarray().flatten()
    elif problem_data_index is not None:
        flat_problem_data = problem_data_tensor @ param_vec
    else:
        param_vec = sp.csc_matrix(param_vec[:, None])
        flat_problem_data = problem_data_tensor @ param_vec


    if problem_data_index is not None:
        indices, indptr, shape = problem_data_index
        M = sp.csc_matrix(
            (flat_problem_data, indices, indptr), shape=shape)
    else:
        n_cols = var_length
        if with_offset:
            n_cols += 1
        M = flat_problem_data.reshape((-1, n_cols), order='F').tocsc()

    if with_offset:
        A = M[:, :-1].tocsc()
        b = np.squeeze(M[:, -1].toarray().flatten())
    else:
        A = M.tocsc()
        b = None

    if nonzero_rows is not None and nonzero_rows.size > 0:
        A_nrows, _ = A.shape
        A_rows, A_cols = nonzero_csc_matrix(A)
        A_vals = np.append(A.data, np.zeros(nonzero_rows.size))
        A_rows = np.append(A_rows, nonzero_rows % A_nrows)
        A_cols = np.append(A_cols, nonzero_rows // A_nrows)
        A = sp.csc_matrix((A_vals, (A_rows, A_cols)),
                                    shape=A.shape)

    return (A, b)


def get_default_canon_backend() -> str:
    """
    Returns the default canonicalization backend, which can be set globally using an
    environment variable.
    """
    return os.environ.get('CVXPY_DEFAULT_CANON_BACKEND', s.DEFAULT_CANON_BACKEND)


def get_problem_matrix(linOps,
                       var_length,
                       id_to_col,
                       param_to_size,
                       param_to_col,
                       constr_length,
                       canon_backend: str | None = None
                       ):
    """
    Builds a sparse representation of the problem data.

    Parameters
    ----------
        linOps: A list of python linOp trees representing an affine expression
        var_length: The total length of the variables.
        id_to_col: A map from variable id to column offset.
        param_to_size: A map from parameter id to parameter size.
        param_to_col: A map from parameter id to column in tensor.
        constr_length: Summed sizes of constraints input.
        canon_backend :
            'CPP' (default) | 'SCIPY'
            Specifies which backend to use for canonicalization, which can affect
            compilation time. Defaults to None, i.e., selecting the default backend.

    Returns
    -------
        A sparse (CSC) matrix with constr_length * (var_length + 1) rows and
        param_size + 1 columns (where param_size is the length of the
        parameter vector).
    """

    # Allow to switch default backends through an environment variable for CI
    default_canon_backend = get_default_canon_backend()
    canon_backend = default_canon_backend if not canon_backend else canon_backend

    if canon_backend == s.CPP_CANON_BACKEND:
        from cvxpy.cvxcore.python.cppbackend import build_matrix
        return build_matrix(id_to_col, param_to_size, param_to_col, var_length, constr_length, linOps)

    elif canon_backend in {s.SCIPY_CANON_BACKEND, s.RUST_CANON_BACKEND,
                           s.NUMPY_CANON_BACKEND}:
        param_size_plus_one = sum(param_to_size.values())
        output_shape = (np.int64(constr_length)*np.int64(var_length+1),
                   param_size_plus_one)
        if len(linOps) > 0:
            backend = CanonBackend.get_backend(canon_backend, id_to_col,
                                                          param_to_size, param_to_col,
                                                          param_size_plus_one, var_length)
            A_py = backend.build_matrix(linOps)
        else:
            A_py = sp.csc_matrix(((), ((), ())), output_shape)
        assert A_py.shape == output_shape
        return A_py
    else:
        raise ValueError(f'Unknown backend: {canon_backend}')

