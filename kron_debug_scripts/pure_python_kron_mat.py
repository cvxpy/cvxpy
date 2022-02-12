from typing import Union

import numpy as np
import scipy.sparse as spar


def get_kronl_mat(lh_dims, rh: Union[np.ndarray, spar.spmatrix], sparse=False):
    """
    Let A be an operator of dimensions "lh_dims", and rh be an ndarray.
    Returns a representation of the parameterized kronecker product operator
    A \mapsto kron(A, rh), where the output out=kron(A, rh) is represented
    by vectorizing in column-major order, and the input A is also represented
    by vectorizing in column-major order.
    """
    lh_rows, lh_cols = lh_dims
    rh_rows, rh_cols = rh.shape

    # build row indices for the first column of mat
    #   Need a column-major vectorized representation of rh
    #   Access such a representation by converting to CSC and then accessing columns.
    kron_rows = lh_rows * rh_rows
    base_row_indices = []
    row_offset = 0
    if isinstance(rh, spar.spmatrix):
        rh = rh.tocsc()
        for rh_col in range(rh_cols):
            nz_rows = rh[:, rh_col].nonzero()[0]
            nz_rows += row_offset
            base_row_indices.append(nz_rows)
            row_offset += kron_rows
        vec_rh = rh.data
    else:
        for rh_col in range(rh_cols):
            block = np.arange(row_offset, row_offset + rh_rows)
            base_row_indices.append(block)
            row_offset += kron_rows
        vec_rh = rh.ravel(order='F')

    base_row_indices = np.concatenate(base_row_indices)
    vec_rh_size = vec_rh.size
    rh_size = rh_cols * rh_rows
    mat_rows = []
    mat_cols = []
    mat_vals = []
    outer_row_offset = 0
    for lh_col in range(lh_cols):
        inner_row_offset = outer_row_offset
        for lh_row in range(lh_rows):
            col = lh_row + lh_col * lh_rows
            mat_cols.append(col * np.ones(vec_rh_size, dtype=int))
            mat_vals.append(vec_rh)
            mat_rows.append(base_row_indices + inner_row_offset)
            inner_row_offset += rh_rows
        outer_row_offset += (lh_rows * rh_size)
    mat_rows = np.concatenate(mat_rows)
    mat_cols = np.concatenate(mat_cols)
    mat_vals = np.concatenate(mat_vals)
    lh_size = lh_rows * lh_cols
    mat = spar.coo_matrix((mat_vals, (mat_rows, mat_cols)),
                          shape=(rh_size * lh_size, lh_size))
    if sparse:
        return mat
    else:
        return mat.A


def check_kronl_mat(lh: np.ndarray, rh: Union[np.ndarray, spar.spmatrix]):
    lh_vec = lh.flatten(order='F')
    mat = get_kronl_mat(lh.shape, rh)
    kron_vec = mat @ lh_vec
    if isinstance(rh, spar.spmatrix):
        rh = rh.A
    expected = np.kron(lh, rh)
    actual = kron_vec.reshape(expected.shape, order='F')
    np.testing.assert_array_almost_equal(actual, expected)


if __name__ == '__main__':
    z_dims = (2, 2)
    for c_dims in [(1, 1), (2, 1), (1, 2), (5, 5), (3, 7), (7, 9), (11, 3)]:
        # use a sparse constant
        check_kronl_mat(np.random.rand(*z_dims), spar.random(*c_dims, density=0.9))
        # use a dense constant
        check_kronl_mat(np.random.rand(*z_dims), spar.random(*c_dims, density=0.9).A)
