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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
import numpy as np
import scipy.sparse as sp


def get_row_nnz(mat, row):
    """Return the number of nonzeros in row.
    """
    return mat.indptr[row+1] - mat.indptr[row]


def compress_matrix(A, b, equil_eps: float = 1e-10):
    """Compresses A and b by eliminating redundant rows.

    Identifies rows that are multiples of another row.
    Reduces A and b to C = PA, d = Pb, where P has one
    nonzero per row.

    Parameters
    ----------
    A : SciPy CSR matrix
        The constraints matrix to compress.
    b : NumPy 1D array
        The vector associated with the constraints matrix.
    equil_eps : float, optional
        Standard for considering two numbers equivalent.

    Returns
    -------
    tuple
        The tuple (A, b, P) where A and b are compressed according to P.
    """
    # Data for compression matrix.
    P_V = []
    P_I = []
    P_J = []
    # List of rows to keep.
    row_to_keep = []
    # A map of sparsity pattern to row list.
    sparsity_to_row = {}
    prev_ptr = A.indptr[0]
    for row_num in range(A.shape[0]):
        keep_row = True
        ptr = A.indptr[row_num+1]
        pattern = tuple(A.indices[prev_ptr:ptr])
        # Eliminate empty rows.
        nnz = ptr - prev_ptr
        if nnz == 0 or np.linalg.norm(A.data[prev_ptr:ptr]) < equil_eps:
            keep_row = False
            P_V.append(0.)
            P_I.append(row_num)
            P_J.append(0)
        # Sparsity pattern is the same or there was a false collision.
        # Check rows have the same number of nonzeros.
        elif pattern in sparsity_to_row and nnz == \
                get_row_nnz(A, sparsity_to_row[pattern][0]):
            # Now test if one row is a multiple of another.
            row_matches = sparsity_to_row[pattern]
            for row_match in row_matches:
                cur_vals = A.data[prev_ptr:ptr]
                prev_match_ptr = A.indptr[row_match]
                match_ptr = A.indptr[row_match+1]
                match_vals = A.data[prev_match_ptr:match_ptr]
                # Ratio should be constant.
                ratio = cur_vals/match_vals
                if np.ptp(ratio) < equil_eps and \
                   abs(ratio[0] - b[row_num]/b[row_match]) < equil_eps:
                    keep_row = False
                    P_V.append(ratio[0])
                    P_I.append(row_num)
                    P_J.append(row_match)
            if keep_row:
                sparsity_to_row[pattern].append(row_num)
        # Pattern doesn't match anything present.
        else:
            sparsity_to_row[pattern] = [row_num]

        if keep_row:
            row_to_keep.append(row_num)
            P_V.append(1.)
            P_I.append(row_num)
            P_J.append(len(row_to_keep)-1)

    # Compress A and b.
    cols = max(len(row_to_keep), 1)
    P = sp.coo_matrix((P_V, (P_I, P_J)), (A.shape[0], cols))
    A_compr = A[row_to_keep, :]
    b_compr = b[row_to_keep]
    return (A_compr, b_compr, P)
