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
#    This file is part of CVXcanon.
#
#    CVXcanon is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    CVXcanon is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with CVXcanon.  If not, see <http:#www.gnu.org/licenses/>.

from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops import lin_utils as lu

import CVXcanon
import numpy as np
import scipy.sparse
from collections import deque


# def atleast_2d_shape(shape):
#     if len(shape) == 0:
#         return (1, 1)
#     elif len(shape) == 1:
#         return shape + (1,)
#     else:
#         return shape


# def atleast_2d_tree(root):
#     if len(root.args) == 0:
#         shape = atleast_2d_shape(root.shape)
#         return lo.LinOp(root.type, shape, [], root.data)

#     prom_args = []
#     for arg in root.args:
#         prom_args.append(atleast_2d_tree(arg))
#     if isinstance(root.data, lo.LinOp):
#         prom_data = atleast_2d_tree(root.data)
#     else:
#         prom_data = root.data

#     if root.type == lo.SUM:
#         shape = prom_args[0].shape
#         args = []
#         for prom_arg in prom_args:
#             if shape != prom_arg.shape:
#                 args.append(lu.transpose(prom_arg))
#             else:
#                 args.append(prom_arg)
#         return lu.sum_expr(args)
#     elif root.type == lo.MUL:
#         lh_arg = prom_data
#         rh_arg = prom_args[0]
#         if lh_arg.shape[1] != rh_arg.shape[0]:
#             # transpose the argument, not the data.
#             rh_arg = lu.transpose(rh_arg)
#         return lu.mul_expr(lh_arg, rh_arg)
#     elif root.type == lo.RMUL:
#         lh_arg = prom_args[0]
#         rh_arg = prom_data
#         if lh_arg.shape[1] != rh_arg.shape[0]:
#             lh_arg = lu.transpose(lh_arg)
#         return lu.rmul_expr(lh_arg, rh_arg)
#     else:
#         shape = atleast_2d_shape(root.shape)
#         return lo.LinOp(root.type, shape, prom_args, prom_data)


def get_problem_matrix(constrs, id_to_col=None, constr_offsets=None):
    """
    Builds a sparse representation of the problem data by calling CVXCanon's
    C++ build_matrix function.

    Parameters
    ----------
        constrs: A list of python linOp trees
        id_to_col: A map from variable id to offset withoun our matrix

    Returns
    ----------
        V, I, J: numpy arrays encoding a sparse representation of our problem
        const_vec: a numpy column vector representing the constant_data in our problem
    """
    linOps = [constr.expr for constr in constrs]
    lin_vec = CVXcanon.LinOpVector()

    id_to_col_C = CVXcanon.IntIntMap()
    if id_to_col is None:
        id_to_col = {}

    # Loading the variable offsets from our
    # Python map into a C++ map
    for id, col in id_to_col.items():
        id_to_col_C[int(id)] = int(col)

    # This array keeps variables data in scope
    # after build_lin_op_tree returns
    tmp = []
    for lin in linOps:
        tree = build_lin_op_tree(lin, tmp)
        tmp.append(tree)
        lin_vec.push_back(tree)

    if constr_offsets is None:
        problemData = CVXcanon.build_matrix(lin_vec, id_to_col_C)
    else:
        # Load constraint offsets into a C++ vector
        constr_offsets_C = CVXcanon.IntVector()
        for offset in constr_offsets:
            constr_offsets_C.push_back(int(offset))
        problemData = CVXcanon.build_matrix(lin_vec, id_to_col_C,
                                            constr_offsets_C)

    # Unpacking
    V = problemData.getV(len(problemData.V))
    I = problemData.getI(len(problemData.I))
    J = problemData.getJ(len(problemData.J))
    const_vec = problemData.getConstVec(len(problemData.const_vec))

    return V, I, J, const_vec.reshape(-1, 1)


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
        return np.asfortranarray(np.matrix(matrix))
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
    else: # TODO remove this case.
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
        vec = CVXcanon.IntVector()
        for var in [sl.start, sl.stop, sl.step]:
            vec.push_back(var)
        linC.slice.push_back(vec)


type_map = {
    "VARIABLE": CVXcanon.VARIABLE,
    "PROMOTE": CVXcanon.PROMOTE,
    "MUL": CVXcanon.MUL,
    "RMUL": CVXcanon.RMUL,
    "MUL_ELEM": CVXcanon.MUL_ELEM,
    "DIV": CVXcanon.DIV,
    "SUM": CVXcanon.SUM,
    "NEG": CVXcanon.NEG,
    "INDEX": CVXcanon.INDEX,
    "TRANSPOSE": CVXcanon.TRANSPOSE,
    "SUM_ENTRIES": CVXcanon.SUM_ENTRIES,
    "TRACE": CVXcanon.TRACE,
    "RESHAPE": CVXcanon.RESHAPE,
    "DIAG_VEC": CVXcanon.DIAG_VEC,
    "DIAG_MAT": CVXcanon.DIAG_MAT,
    "UPPER_TRI": CVXcanon.UPPER_TRI,
    "CONV": CVXcanon.CONV,
    "HSTACK": CVXcanon.HSTACK,
    "VSTACK": CVXcanon.VSTACK,
    "SCALAR_CONST": CVXcanon.SCALAR_CONST,
    "DENSE_CONST": CVXcanon.DENSE_CONST,
    "SPARSE_CONST": CVXcanon.SPARSE_CONST,
    "NO_OP": CVXcanon.NO_OP,
    "KRON": CVXcanon.KRON
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
    root_linC = CVXcanon.LinOp()
    Q.append((root_linPy, root_linC))

    while len(Q) > 0:
        linPy, linC = Q.popleft()

        # Updating the arguments our LinOp
        for argPy in linPy.args:
            tree = CVXcanon.LinOp()
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
        elif isinstance(linPy.data, float) or isinstance(linPy.data, int):
            linC.set_dense_data(format_matrix(linPy.data, format='scalar'))
            linC.data_ndim = 0
        # data is supposed to be a LinOp
        elif isinstance(linPy.data, lo.LinOp) and linPy.data.type == 'scalar_const':
            linC.set_dense_data(format_matrix(linPy.data.data, format='scalar'))
            linC.data_ndim = 0
        else:
            set_matrix_data(linC, linPy)

    return root_linC
