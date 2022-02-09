from typing import Union

import numpy as np
import scipy.sparse as spar

import cvxpy as cp

"""
kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
             [M[1,0] * N   , ..., M[1, end] * N  ]
             ...
             [M[end, 0] * N, ..., M[end, end] * N] 
"""


def random_test(z_dims, c_dims, left=True, param=True, seed=0):
    #  Z, C, U, prob = random_test((1, 1), (2, 2))
    #  G = prob.get_problem_data(solver='ECOS')[0]['G'].A
    Z = cp.Variable(shape=z_dims)
    np.random.seed(seed)
    C_value = np.random.rand(*c_dims).round(decimals=2)
    if param:
        C = cp.Parameter(shape=c_dims)
        C.value = C_value
    else:
        C = cp.Constant(C_value)
    L = np.random.rand(*Z.shape)
    U = L + np.random.rand(*Z.shape)
    U = U.round(decimals=2)
    if left:
        constraints = [cp.kron(U, C) >= cp.kron(Z, C), cp.kron(Z, C) >= cp.kron(L, C)]
    else:
        constraints = [cp.kron(C, U) >= cp.kron(C, Z), cp.kron(C, Z) >= cp.kron(C, L)]
    obj_expr = cp.sum(Z)

    prob = cp.Problem(cp.Maximize(obj_expr), constraints)
    prob.solve(solver='ECOS')
    Z_actual = Z.value
    Z_expect = U
    if Z_actual is not None:
        print(np.allclose(Z_actual, Z_expect))
    else:
        print(False)
        #raise RuntimeError()

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    prob.solve(solver='ECOS')
    Z_actual = Z.value
    Z_expect = L
    if Z_actual is not None:
        print(np.allclose(Z_actual, Z_expect))
    else:
        print(False)
       # raise RuntimeError()

    return Z, C, U, prob


def label_columns(prob):
    variables = prob.variables()
    idx = 0
    label_cons = []
    labels = []
    for v in variables:
        cur_labels = np.arange(idx, idx + v.size).reshape(v.shape, order='F')
        labels.append((cur_labels, v))
        label_cons.append(cp.multiply(cur_labels, v) <= 0)
        idx += v.size
    cons = label_cons + prob.constraints
    prob2 = cp.Problem(prob.objective, cons)
    return prob2, labels


def kron_mat(lh_dims, rh: Union[np.ndarray, spar.spmatrix], sparse=False):
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


def check_kron_mat(lh: np.ndarray, rh: Union[np.ndarray, spar.spmatrix]):
    lh_vec = lh.flatten(order='F')
    mat = kron_mat(lh.shape, rh)
    kron_vec = mat @ lh_vec
    if isinstance(rh, spar.spmatrix):
        rh = rh.A
    expected = np.kron(lh, rh)
    actual = kron_vec.reshape(expected.shape, order='F')
    np.testing.assert_array_almost_equal(actual, expected)


if __name__ == '__main__':
    seed = 0
    z_dims = (2, 2)
    #for c_dims in [(5, 5), (3, 7), (7, 9), (11, 3)]:
    #    check_kron_mat(np.random.rand(*z_dims), spar.random(*c_dims, density=0.9))
    #    check_kron_mat(np.random.rand(*z_dims), spar.random(*c_dims, density=0.9).A)
    #for c_dims in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 7), (7, 3)]:
    #    random_test(z_dims, c_dims, seed=0)
    c_dims = (1, 1)

    LEFT = False
    Z_f, C_f, U_f, prob_f0 = random_test(z_dims, c_dims, left=LEFT, param=False, seed=0)
    prob_f1, _ = label_columns(prob_f0)
    data_f = prob_f0.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
    Gf = data_f['G'].A
    hf = data_f['h']

    Z_t, C_t, U_t, prob_t0 = random_test(z_dims, c_dims, left=LEFT, param=True, seed=0)
    prob_t1, _ = label_columns(prob_t0)
    data_t = prob_t0.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
    Gt = data_t['G'].A
    ht = data_t['h']

    print()
    print(Gf)
    print(hf)
    print()
    print(Gt)
    print(ht)
