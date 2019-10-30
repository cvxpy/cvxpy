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
from cvxpy.lin_ops import lin_op as lo
import cvxpy.cvxcore.python.cvxcore as cvxcore
import numbers
import numpy as np
import scipy.sparse
from collections import deque


def get_parameter_vector(param_size,
                         param_id_to_col,
                         param_id_to_size,
                         param_id_to_value_fn,
                         zero_offset=False):
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
    # TODO handle parameters with structure.
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


def get_matrix_and_offset_from_tensor(problem_data_tensor, param_vec,
                                      var_length, nonzero_rows=None):
    """Applies problem_data_tensor to param_vec to obtain matrix, offset

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

    Returns
    -------
        A tuple (A, b), where A is a matrix with `var_length` columns
        and b is a flattened NumPy array representing the constant offset.
    """
    if param_vec is None:
        tensor_application = problem_data_tensor
    else:
        if scipy.sparse.issparse(problem_data_tensor):
            param_vec = scipy.sparse.csc_matrix(param_vec[:, None])
        tensor_application = problem_data_tensor @ param_vec
    A_concat_b = tensor_application.reshape(
        (-1, var_length + 1), order='F').tocsc()

    A = A_concat_b[:, :-1].tocsc()
    if nonzero_rows is not None and nonzero_rows.size > 0:
        A_nrows, _ = A.shape
        A_rows, A_cols = nonzero_csc_matrix(A)
        A_vals = np.append(A.data, np.zeros(nonzero_rows.size))
        A_rows = np.append(A_rows, nonzero_rows % A_nrows)
        A_cols = np.append(A_cols, nonzero_rows // A_nrows)
        A = scipy.sparse.csc_matrix((A_vals, (A_rows, A_cols)),
                                    shape=A.shape)

    b = np.squeeze(A_concat_b[:, -1].toarray().flatten())
    return (A, b)


def get_matrix_and_offset_from_unparameterized_tensor(problem_data_tensor,
                                                      var_length):
    """Converts unparameterized tensor to matrix offset representation

    problem_data_tensor _must_ have been obtained from calling
    get_problem_matrix on a problem with 0 parameters.

    Parameters
    ----------
        problem_data_tensor: tensor returned from get_problem_matrix,
            representing an affine map
        var_length: the number of variables

    Returns
    -------
        A tuple (A, b), where A is a matrix with `var_length` columns
        and b is a flattened NumPy array representing the constant offset.
    """
    assert problem_data_tensor.shape[1] == 1
    return get_matrix_and_offset_from_tensor(
        problem_data_tensor, None, var_length)


def get_problem_matrix(linOps,
                       var_length,
                       id_to_col,
                       param_to_size,
                       param_to_col,
                       constr_length):
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

    Returns
    -------
        A sparse (CSC) matrix with constr_length * (var_length + 1) rows and
        param_size + 1 columns (where param_size is the length of the
        parameter vector).
    """
    lin_vec = cvxcore.ConstLinOpVector()

    id_to_col_C = cvxcore.IntIntMap()
    for id, col in id_to_col.items():
        id_to_col_C[int(id)] = int(col)

    param_to_size_C = cvxcore.IntIntMap()
    for id, size in param_to_size.items():
        param_to_size_C[int(id)] = int(size)

    # dict to memoize construction of C++ linOps, and to keep Python references
    # to them to prevent their deletion
    linPy_to_linC = {}
    for lin in linOps:
        build_lin_op_tree(lin, linPy_to_linC)
        tree = linPy_to_linC[lin]
        lin_vec.push_back(tree)
    problemData = cvxcore.build_matrix(lin_vec,
                                       int(var_length),
                                       id_to_col_C,
                                       param_to_size_C)

    # Populate tensors with info from problemData.
    tensor_V = {}
    tensor_I = {}
    tensor_J = {}
    for param_id, size in param_to_size.items():
        tensor_V[param_id] = []
        tensor_I[param_id] = []
        tensor_J[param_id] = []
        problemData.param_id = param_id
        for i in range(size):
            problemData.vec_idx = i
            prob_len = problemData.getLen()
            tensor_V[param_id].append(problemData.getV(prob_len))
            tensor_I[param_id].append(problemData.getI(prob_len))
            tensor_J[param_id].append(problemData.getJ(prob_len))

    # Reduce tensors to a single sparse CSR matrix.
    V = []
    I = []
    J = []
    # one of the 'parameters' in param_to_col is a constant scalar offset,
    # hence 'plus_one'
    param_size_plus_one = 0
    for param_id, col in param_to_col.items():
        size = param_to_size[param_id]
        param_size_plus_one += size
        for i in range(size):
            V.append(tensor_V[param_id][i])
            I.append(tensor_I[param_id][i] +
                     tensor_J[param_id][i]*constr_length)
            J.append(tensor_J[param_id][i]*0 + (i + col))
    V = np.concatenate(V)
    I = np.concatenate(I)
    J = np.concatenate(J)
    A = scipy.sparse.csc_matrix(
        (V, (I, J)), shape=(constr_length*(var_length+1), param_size_plus_one))
    return A


def format_matrix(matrix, shape=None, format='dense'):
    """Returns the matrix in the appropriate form for SWIG wrapper"""
    if (format == 'dense'):
        # Ensure is 2D.
        if len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = shape + (1,)
        return np.reshape(matrix, shape, order='F')
    elif(format == 'sparse'):
        return scipy.sparse.coo_matrix(matrix)
    elif(format == 'scalar'):
        return np.asfortranarray([[matrix]])
    else:
        raise NotImplementedError()


TYPE_MAP = {
    "VARIABLE": cvxcore.VARIABLE,
    "PARAM": cvxcore.PARAM,
    "PROMOTE": cvxcore.PROMOTE,
    "MUL": cvxcore.MUL,
    "RMUL": cvxcore.RMUL,
    "MUL_ELEM": cvxcore.MUL_ELEM,
    "DIV": cvxcore.DIV,
    "SUM": cvxcore.SUM,
    "NEG": cvxcore.NEG,
    "INDEX": cvxcore.INDEX,
    "TRANSPOSE": cvxcore.TRANSPOSE,
    "SUM_ENTRIES": cvxcore.SUM_ENTRIES,
    "TRACE": cvxcore.TRACE,
    "RESHAPE": cvxcore.RESHAPE,
    "DIAG_VEC": cvxcore.DIAG_VEC,
    "DIAG_MAT": cvxcore.DIAG_MAT,
    "UPPER_TRI": cvxcore.UPPER_TRI,
    "CONV": cvxcore.CONV,
    "HSTACK": cvxcore.HSTACK,
    "VSTACK": cvxcore.VSTACK,
    "SCALAR_CONST": cvxcore.SCALAR_CONST,
    "DENSE_CONST": cvxcore.DENSE_CONST,
    "SPARSE_CONST": cvxcore.SPARSE_CONST,
    "NO_OP": cvxcore.NO_OP,
    "KRON": cvxcore.KRON
}


def get_type(linPy):
    ty = linPy.type.upper()
    if ty in TYPE_MAP:
        return TYPE_MAP[ty]
    else:
        raise NotImplementedError("Type %s is not supported." % ty)


def set_matrix_data(linC, linPy):
    """Calls the appropriate cvxcore function to set the matrix data field of
       our C++ linOp.
    """
    if get_type(linPy) == cvxcore.SPARSE_CONST:
        coo = format_matrix(linPy.data, format='sparse')
        linC.set_sparse_data(coo.data, coo.row.astype(float),
                             coo.col.astype(float), coo.shape[0],
                             coo.shape[1])
    else:
        linC.set_dense_data(format_matrix(linPy.data, shape=linPy.shape))
        linC.set_data_ndim(len(linPy.data.shape))


def set_slice_data(linC, linPy):
    """
    Loads the slice data, start, stop, and step into our C++ linOp.
    The semantics of the slice operator is treated exactly the same as in
    Python.  Note that the 'None' cases had to be handled at the wrapper level,
    since we must load integers into our vector.
    """
    for i, sl in enumerate(linPy.data):
        slice_vec = cvxcore.IntVector()
        for var in [sl.start, sl.stop, sl.step]:
            slice_vec.push_back(int(var))
        linC.push_back_slice_vec(slice_vec)


def set_linC_data(linC, linPy):
    """Sets numerical data fields in linC."""
    assert linPy.data is not None
    if isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
        slice_data = set_slice_data(linC, linPy)
    elif isinstance(linPy.data, float) or isinstance(linPy.data,
                                                   numbers.Integral):
        linC.set_dense_data(format_matrix(linPy.data, format='scalar'))
        linC.set_data_ndim(0)
    else:
        set_matrix_data(linC, linPy)


def make_linC_from_linPy(linPy, linPy_to_linC):
    """Construct a C++ LinOp corresponding to LinPy.

    Children of linPy are retrieved from linPy_to_linC.
    """
    if linPy in linPy_to_linC:
        return
    typ = get_type(linPy)
    shape = cvxcore.IntVector()
    lin_args_vec = cvxcore.ConstLinOpVector()
    for dim in linPy.shape:
        shape.push_back(int(dim))
    for argPy in linPy.args:
        lin_args_vec.push_back(linPy_to_linC[argPy])
    linC = cvxcore.LinOp(typ, shape, lin_args_vec)
    linPy_to_linC[linPy] = linC

    if linPy.data is not None:
        if isinstance(linPy.data, lo.LinOp):
            linC_data = linPy_to_linC[linPy.data]
            linC.set_linOp_data(linC_data)
            linC.set_data_ndim(len(linPy.data.shape))
        else:
            set_linC_data(linC, linPy)


def build_lin_op_tree(root_linPy, linPy_to_linC):
    """Construct C++ LinOp tree from Python LinOp tree.

    Constructed C++ linOps are stored in the linPy_to_linC dict,
    which maps Python linOps to their corresponding C++ linOps.

    Parameters
    ----------
        linPy_to_linC: a dict for memoizing construction and storing
            the C++ LinOps
    """
    bfs_stack = [root_linPy]
    post_order_stack = []
    while bfs_stack:
        linPy = bfs_stack.pop()
        if linPy not in linPy_to_linC:
            post_order_stack.append(linPy)
            for arg in linPy.args:
                bfs_stack.append(arg)
            if isinstance(linPy.data, lo.LinOp):
                bfs_stack.append(linPy.data)
    while post_order_stack:
        linPy = post_order_stack.pop()
        make_linC_from_linPy(linPy, linPy_to_linC)
