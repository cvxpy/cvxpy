"""
Copyright, the CVXPY authors

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

import numbers
from typing import List

import numpy as np
import scipy.sparse as sp

import cvxpy.lin_ops.lin_op as lo
import cvxpy.settings as s

try:
    import cvxpy.cvxcore.python.cvxcore as cvxcore
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Tried using the C++ backend, but the cvxcore module was not installed."
    )


def build_matrix(
    id_to_col: dict,
    param_to_size: dict,
    param_to_col: dict,
    var_length: int,
    constr_length: int,
    linOps: List[lo.LinOp],
) -> sp.csc_matrix:
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

    problemData = cvxcore.build_matrix(
        lin_vec, int(var_length), id_to_col_C, param_to_size_C, s.get_num_threads()
    )

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
            I.append(tensor_I[param_id][i] + tensor_J[param_id][i] * constr_length)
            J.append(tensor_J[param_id][i] * 0 + (i + col))
    V = np.concatenate(V)
    I = np.concatenate(I)
    J = np.concatenate(J)

    output_shape = (
        np.int64(constr_length) * np.int64(var_length + 1),
        param_size_plus_one,
    )
    A = sp.csc_matrix((V, (I, J)), shape=output_shape)
    return A


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
    "KRON_R": cvxcore.KRON_R,
    "KRON_L": cvxcore.KRON_L,
}


def get_type(linPy):
    """
    Returns the cvxcore type corresponding to the type of linPy.
    """

    ty = linPy.type.upper()
    if ty in TYPE_MAP:
        return TYPE_MAP[ty]
    else:
        raise NotImplementedError(f"Type {ty} is not supported.")


def set_linC_data(linC, linPy) -> None:
    """Sets numerical data fields in linC."""
    assert linPy.data is not None
    if isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
        set_slice_data(linC, linPy)
    elif isinstance(linPy.data, float) or isinstance(linPy.data, numbers.Integral):
        linC.set_dense_data(format_matrix(linPy.data, format="scalar"))
        linC.set_data_ndim(0)
    else:
        set_matrix_data(linC, linPy)


def make_linC_from_linPy(linPy, linPy_to_linC) -> None:
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


def set_slice_data(linC, linPy) -> None:
    """
    Loads the slice data, start, stop, and step into our C++ linOp.
    The semantics of the slice operator is treated exactly the same as in
    Python.  Note that the 'None' cases had to be handled at the wrapper level,
    since we must load integers into our vector.
    """

    for sl in linPy.data:
        slice_vec = cvxcore.IntVector()
        for var in [sl.start, sl.stop, sl.step]:
            slice_vec.push_back(int(var))
        linC.push_back_slice_vec(slice_vec)


def build_lin_op_tree(root_linPy, linPy_to_linC) -> None:
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


def set_matrix_data(linC, linPy) -> None:
    """Calls the appropriate cvxcore function to set the matrix data field of
    our C++ linOp.
    """

    if get_type(linPy) == cvxcore.SPARSE_CONST:
        coo = format_matrix(linPy.data, format="sparse")
        linC.set_sparse_data(
            coo.data,
            coo.row.astype(float),
            coo.col.astype(float),
            coo.shape[0],
            coo.shape[1],
        )
    else:
        linC.set_dense_data(format_matrix(linPy.data, shape=linPy.shape))
        linC.set_data_ndim(len(linPy.data.shape))


def format_matrix(matrix, shape=None, format="dense"):
    """Returns the matrix in the appropriate form for SWIG wrapper"""
    if format == "dense":
        # Ensure is 2D.
        if len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = shape + (1,)
        return np.reshape(matrix, shape, order="F")
    elif format == "sparse":
        return sp.coo_matrix(matrix)
    elif format == "scalar":
        return np.asfortranarray([[matrix]])
    else:
        raise NotImplementedError()
