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
import scipy.sparse as sp
import numpy as np

def get_row_nnz(mat, row):
    """Return the number of nonzeros in row.
    """
    return mat.indptr[row+1] - mat.indptr[row]

def compress_matrix(A, b, equil_eps=1e-10):
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
    # SciPy 0.13 can't index using an empty list.
    if len(row_to_keep) == 0:
        A_compr = A[0:0,:]
        b_compr = b[0:0,:]
    else:
        A_compr = A[row_to_keep,:]
        b_compr = b[row_to_keep,:]
    return (A_compr, b_compr, P)
