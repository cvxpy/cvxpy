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


def get_problem_matrix(linOps,
                       var_length,
                       id_to_col,
                       param_to_size,
                       param_to_col,
                       constr_length,
                       constr_offsets=None):
    """
    Builds a sparse representation of the problem data by calling CVXCanon's
    C++ build_matrix function.

    Parameters
    ----------
        linOps: A list of python linOp trees
        var_length: The total length of the variables.
        id_to_col: A map from variable id to column offset.
        param_to_size: A map from parameter id to parameter size.
        param_to_col: A map from parameter id to column in tensor.
        constr_offsets: A map from constraint id to row offset.

    Returns
    ----------
        V, I, J: numpy arrays encoding a sparse representation of our problem
        const_vec: a numpy column vector representing the constant_data in our problem
    """
    lin_vec = cvxcore.LinOpVector()

    # Loading the variable offsets from our
    # Python map into a C++ map
    id_to_col_C = cvxcore.IntIntMap()
    for id, col in id_to_col.items():
        id_to_col_C[int(id)] = int(col)

    # Loading the param_to_size from our
    # Python map into a C++ map
    param_to_size_C = cvxcore.IntIntMap()
    for id, size in param_to_size.items():
        param_to_size_C[int(id)] = int(size)

    # This array keeps variables data in scope
    # after build_lin_op_tree returns
    tmp = []
    for lin in linOps:
        tree = build_lin_op_tree(lin, tmp)
        lin_vec.push_back(tree)

    if constr_offsets is None:
        problemData = cvxcore.build_matrix(lin_vec,
                                           int(var_length),
                                           id_to_col_C,
                                           param_to_size_C)
    else:
        # Load constraint offsets into a C++ vector
        constr_offsets_C = cvxcore.IntVector()
        for offset in constr_offsets:
            constr_offsets_C.push_back(int(offset))
        problemData = cvxcore.build_matrix(lin_vec,
                                           var_length,
                                           id_to_col_C,
                                           param_to_size_C,
                                           constr_offsets_C)

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
    total_size = 0
    for param_id, col in param_to_col.items():
        size = param_to_size[param_id]
        total_size += size
        for i in range(size):
            V.append(tensor_V[param_id][i])
            I.append(tensor_I[param_id][i] +
                     tensor_J[param_id][i]*constr_length)
            J.append(tensor_J[param_id][i]*0 + (i + col))
    V = np.concatenate(V)
    I = np.concatenate(I)
    J = np.concatenate(J)
    A = scipy.sparse.csc_matrix((V, (I, J)),
                                shape=(constr_length*(var_length+1),
                                       total_size))
    return A


def format_matrix(matrix, shape=None, format='dense'):
    """ Returns the matrix in the appropriate form,
        so that it can be efficiently loaded with our swig wrapper
    """
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


def set_matrix_data(linC, linPy):
    """Calls the appropriate CVXCanon function to set the matrix data field of
       our C++ linOp.
    """
    # data is supposed to be a LinOp
    if isinstance(linPy.data, lo.LinOp):
        if linPy.data.type == 'sparse_const':
            coo = format_matrix(linPy.data.data, format='sparse')
            linC.set_sparse_data(coo.data, coo.row.astype(float),
                                 coo.col.astype(float), coo.shape[0],
                                 coo.shape[1])
        elif linPy.data.type == 'dense_const':
            linC.set_dense_data(format_matrix(linPy.data.data,
                                              shape=linPy.data.shape))
            linC.data_ndim = len(linPy.data.shape)
        else:
            raise NotImplementedError()
    else:  # TODO remove this case.
        if linPy.type == 'sparse_const':
            coo = format_matrix(linPy.data, format='sparse')
            linC.set_sparse_data(coo.data, coo.row.astype(float),
                                 coo.col.astype(float), coo.shape[0],
                                 coo.shape[1])
        else:
            linC.set_dense_data(format_matrix(linPy.data,
                                              shape=linPy.shape))
            linC.data_ndim = len(linPy.data.shape)


def set_slice_data(linC, linPy):
    """
    Loads the slice data, start, stop, and step into our C++ linOp.
    The semantics of the slice operator is treated exactly the same as in
    Python.  Note that the 'None' cases had to be handled at the wrapper level,
    since we must load integers into our vector.
    """
    for i, sl in enumerate(linPy.data):
        vec = cvxcore.IntVector()
        for var in [sl.start, sl.stop, sl.step]:
            vec.push_back(int(var))
        linC.slice.push_back(vec)


type_map = {
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


def get_type(ty):
    if ty in type_map:
        return type_map[ty]
    else:
        raise NotImplementedError("Type %s is not supported." % ty)


def build_lin_op_tree(root_linPy, tmp):
    """
    Breadth-first, pre-order traversal on the Python linOp tree
    Parameters
    -------------
    root_linPy: a Python LinOp tree

    tmp: an array to keep data from going out of scope

    Returns
    --------
    root_linC: a C++ LinOp tree created through our swig interface
    """
    Q = deque()
    root_linC = cvxcore.LinOp()
    Q.append((root_linPy, root_linC))

    while len(Q) > 0:
        linPy, linC = Q.popleft()

        # Updating the arguments our LinOp
        for argPy in linPy.args:
            tree = cvxcore.LinOp()
            tmp.append(tree)
            Q.append((argPy, tree))
            linC.args.push_back(tree)

        # Setting the type of our lin op
        linC.type = get_type(linPy.type.upper())

        # Setting size
        for dim in linPy.shape:
            linC.size.push_back(int(dim))

        # Loading the problem data into the appropriate array format
        if linPy.data is None:
            pass
        elif isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
            set_slice_data(linC, linPy)
        elif isinstance(linPy.data, float) or isinstance(linPy.data, numbers.Integral):
            linC.set_dense_data(format_matrix(linPy.data, format='scalar'))
            linC.data_ndim = 0
        elif isinstance(linPy.data, lo.LinOp):
            # Recurse on LinOp.
            linC_data = build_lin_op_tree(linPy.data, tmp)
            linC.set_linOp_data(linC_data)
        else:
            set_matrix_data(linC, linPy)

    tmp.append(root_linC)
    return root_linC
