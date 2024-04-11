"""

Copyright 2016 Jaehyun Park, 2017 Robin Verschueren, 2017 Akshay Agrawal

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

from __future__ import annotations, division

import operator
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp

from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.lin_op import NO_OP, LinOp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (
    replace_quad_forms,
    restore_quad_forms,
)


@dataclass
class COOData:
    """
    Data for constructing a COO matrix representing a 3d tensor.
    Unlike a in a 2d COO matrix, data is a 2d matrix, with the columns
    representing the parameter slices given by param_idxs.
    """
    data: np.ndarray
    row: np.ndarray
    col: np.ndarray
    shape: tuple[int, int]
    param_idxs: np.ndarray

    def __add__(self, other: COOData) -> COOData:
        # Concatenation becomes addition when constructing
        # COO matrix because repeated indices are summed.

        if self.shape != other.shape:
            raise ValueError("Shapes must match for addition.")
        data = np.concatenate([self.data, other.data], axis=1)
        row = np.concatenate([self.row, other.row], axis=0)
        col = np.concatenate([self.col, other.col], axis=0)
        param_idxs = np.concatenate([self.param_idxs, other.param_idxs], axis=0)
        return COOData(data, row, col, self.shape, param_idxs)

    @classmethod
    def empty_with_shape(cls, shape: tuple[int, int]) -> COOData:
        return COOData(
            np.array([]),
            np.array([], dtype=int),
            np.array([], dtype=int),
            shape,
            np.array([], dtype=int),
        )

# TODO find best format for sparse matrices: csr, csc, dok, lil, ...
class CoeffExtractor:

    def __init__(self, inverse_data, canon_backend: str | None) -> None:
        self.id_map = inverse_data.var_offsets
        self.x_length = inverse_data.x_length
        self.var_shapes = inverse_data.var_shapes
        self.param_shapes = inverse_data.param_shapes
        self.param_to_size = inverse_data.param_to_size
        self.param_id_map = inverse_data.param_id_map
        self.canon_backend = canon_backend

    def get_coeffs(self, expr):
        if expr.is_constant():
            return self.constant(expr)
        elif expr.is_affine():
            return self.affine(expr)
        elif expr.is_quadratic():
            return self.quad_form(expr)
        else:
            raise Exception("Unknown expression type %s." % type(expr))

    def constant(self, expr):
        size = expr.size
        return sp.csr_matrix((size, self.N)), np.reshape(expr.value, (size,),
                                                         order='F')

    def affine(self, expr):
        """Extract problem data tensor from an expression that is reducible to
        A*x + b.

        Applying the tensor to a flattened parameter vector and reshaping
        will recover A and b (see the helpers in canonInterface).

        Parameters
        ----------
        expr : Expression or list of Expressions.
            The expression(s) to process.

        Returns
        -------
        SciPy CSR matrix
            Problem data tensor, of shape
            (constraint length * (variable length + 1), parameter length + 1)
        """
        if isinstance(expr, list):
            expr_list = expr
        else:
            expr_list = [expr]
        assert all([e.is_dpp() for e in expr_list])
        num_rows = sum([e.size for e in expr_list])
        op_list = [e.canonical_form[0] for e in expr_list]
        return canonInterface.get_problem_matrix(op_list,
                                                 self.x_length,
                                                 self.id_map,
                                                 self.param_to_size,
                                                 self.param_id_map,
                                                 num_rows,
                                                 self.canon_backend)

    def extract_quadratic_coeffs(self, affine_expr, quad_forms):
        """ Assumes quadratic forms all have variable arguments.
            Affine expressions can be anything.
        """
        assert affine_expr.is_dpp()
        # Here we take the problem objective, replace all the SymbolicQuadForm
        # atoms with variables of the same dimensions.
        # We then apply the canonInterface to reduce the "affine head"
        # of the expression tree to a coefficient vector c and constant offset d.
        # Because the expression is parameterized, we extend that to a matrix
        # [c1 c2 ...]
        # [d1 d2 ...]
        # where ci,di are the vector and constant for the ith parameter.
        affine_id_map, affine_offsets, x_length, affine_var_shapes = \
            InverseData.get_var_offsets(affine_expr.variables())
        op_list = [affine_expr.canonical_form[0]]
        param_coeffs = canonInterface.get_problem_matrix(op_list,
                                                         x_length,
                                                         affine_offsets,
                                                         self.param_to_size,
                                                         self.param_id_map,
                                                         affine_expr.size,
                                                         self.canon_backend)

        # Iterates over every entry of the parameters vector,
        # and obtains the Pi and qi for that entry i.
        # These are then combined into matrices [P1.flatten(), P2.flatten(), ...]
        # and [q1, q2, ...]
        constant = param_coeffs[-1, :]
        # TODO keep sparse.
        c = param_coeffs[:-1, :].toarray()
        num_params = param_coeffs.shape[1]

        # coeffs stores the P and q for each quad_form,
        # as well as for true variable nodes in the objective.
        coeffs = {}
        # The goal of this loop is to appropriately multiply
        # the matrix P of each quadratic term by the coefficients
        # in param_coeffs. Later we combine all the quadratic terms
        # to form a single matrix P.
        for var in affine_expr.variables():
            # quad_forms maps the ids of the SymbolicQuadForm atoms
            # in the objective to (modified parent node of quad form,
            #                      argument index of quad form,
            #                      quad form atom)
            if var.id in quad_forms:
                # This was a dummy variable
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                c_part = c[var_offset:var_offset+var_size, :]

                # Convert to sparse matrix.
                P = quad_forms[var_id][2].P
                assert (
                    P.value is not None
                ), "P matrix must be instantiated before calling extract_quadratic_coeffs."
                if sp.issparse(P) and not isinstance(P, sp.coo_matrix):
                    P = P.value.tocoo()
                else:
                    P = sp.coo_matrix(P.value)

                # We multiply the columns of P, by c_part
                # by operating directly on the data.
                if var_size > 1:
                    # Multiple quad forms in the same expression, sharing P
                    # TODO remove zeros from data.
                    data = P.data[:, None] * c_part[P.col]
                    param_idxs = np.arange(num_params)
                else:
                    # Eliminate zeros from data by tracking
                    # which indices of the global parameter vector are used.

                    nonzero_idxs = c_part[0] != 0
                    data = P.data[:, None] * c_part[:, nonzero_idxs]
                    param_idxs = np.arange(num_params)[nonzero_idxs]
                P_tup = COOData(data, P.row, P.col, P.shape, param_idxs)
                # Conceptually similar to
                # P = P[:, :, None] * c_part[None, :, :]
                if orig_id in coeffs:
                    if 'P' in coeffs[orig_id]:
                        coeffs[orig_id]['P'] =  coeffs[orig_id]['P'] + P_tup
                    else:
                        coeffs[orig_id]['P'] = P_tup
                else:
                    # No q for dummy variables.
                    coeffs[orig_id] = dict()
                    coeffs[orig_id]['P'] = P_tup
                    shape = (P.shape[0], c.shape[1])
                    if num_params == 1:
                        # Fast path for no parameters, keep q dense.
                        coeffs[orig_id]['q'] = np.zeros(shape)
                    else:
                        coeffs[orig_id]['q'] = sp.coo_matrix(([], ([], [])), shape=shape) 
            else:
                # This was a true variable, so it can only have a q term.
                var_offset = affine_id_map[var.id][0]
                var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                if var.id in coeffs:
                    # Fast path for no parameters, q is dense and so is c.
                    if num_params == 1:
                        coeffs[var.id]['q'] += c[var_offset:var_offset+var_size, :]
                    else:
                        coeffs[var.id]['q'] += param_coeffs[var_offset:var_offset+var_size, :]
                else:
                    coeffs[var.id] = dict()
                    # Fast path for no parameters, q is dense and so is c.
                    if num_params == 1:
                        coeffs[var.id]['q'] = c[var_offset:var_offset+var_size, :]
                    else:
                        coeffs[var.id]['q'] = param_coeffs[var_offset:var_offset+var_size, :]
        return coeffs, constant

    def quad_form(self, expr):
        """Extract quadratic, linear constant parts of a quadratic objective.
        """
        # Insert no-op such that root is never a quadratic form, for easier
        # processing
        root = LinOp(NO_OP, expr.shape, [expr], [])

        # Replace quadratic forms with dummy variables.
        quad_forms = replace_quad_forms(root, {})

        # Calculate affine parts and combine them with quadratic forms to get
        # the coefficients.
        coeffs, constant = self.extract_quadratic_coeffs(root.args[0],
                                                         quad_forms)
        # Restore expression.
        restore_quad_forms(root.args[0], quad_forms)

        # Sort variables corresponding to their starting indices, in ascending
        # order.
        offsets = sorted(self.id_map.items(), key=operator.itemgetter(1))

        # Extract quadratic matrices and vectors
        num_params = constant.shape[1]
        P_list = []
        q_list = []
        P_height = 0
        P_entries = 0
        for var_id, _ in offsets:
            shape = self.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            if var_id in coeffs and 'P' in coeffs[var_id]:
                P = coeffs[var_id]['P']
                P_entries += P.data.size
            else:
                P = COOData.empty_with_shape((size, size))
            if var_id in coeffs and 'q' in coeffs[var_id]:
                q = coeffs[var_id]['q']
            else:
                # Fast path for no parameters.
                if num_params == 1:
                    q = np.zeros((size, num_params))
                else:
                    q = sp.coo_matrix(([], ([], [])), (size, num_params))

            P_list.append(P)
            q_list.append(q)
            P_height += size

        if P_height != self.x_length:
            raise RuntimeError("Resulting quadratic form does not have "
                               "appropriate dimensions")

        # Stitch together Ps and qs and constant.
        P = self.merge_P_list(P_list, P_entries, P_height, num_params)
        q = self.merge_q_list(q_list, constant, num_params)
        return P, q

    def merge_P_list(
            self, 
            P_list: List[COOData],
            P_entries: int, 
            P_height: int, 
            num_params: int,
        ) -> sp.coo_matrix:
        """Conceptually we build a block diagonal matrix
           out of all the Ps, then flatten the first two dimensions.
           eg P1
                P2
           We do this by extending each P with zero blocks above and below.

        Args:
            P_list: list of P submatrices as COOData objects.
            P_entries: number of entries in the merged P matrix.
            P_height: number of rows in the merged P matrix.
            num_params: number of parameters in the problem.
        
        Returns:
            A COO sparse representation of the merged P matrix.
        """
        gap_above = np.int64(0)
        acc_height = np.int64(0)
        rows = np.zeros(P_entries, dtype=np.int64)
        cols = np.zeros(P_entries, dtype=np.int64)
        vals = np.zeros(P_entries)
        entry_offset = 0
        for P in P_list:
            """Conceptually, the code is equivalent to
            ```
            above = np.zeros((gap_above, P.shape[1], num_params))
            below = np.zeros((gap_below, P.shape[1], num_params))
            padded_P = np.concatenate([above, P, below], axis=0)
            padded_P = np.reshape(padded_P, (P_height*P.shape[1], num_params),
                                  order='F')
            padded_P_list.append(padded_P)
            ```
            but done by constructing a COO matrix.
            """
            if P.data.size > 0:
                vals[entry_offset:entry_offset + P.data.size] = P.data.flatten(
                    order='F'
                )
                P_cols_ext = P.col.astype(np.int64) * np.int64(P_height)
                base_rows = gap_above + acc_height + P.row + P_cols_ext
                full_rows = np.tile(base_rows, P.data.size // P.row.size)
                rows[entry_offset:entry_offset + P.data.size] = full_rows
                full_cols = np.repeat(P.param_idxs, P.data.size // P.param_idxs.size)
                cols[entry_offset:entry_offset + P.data.size] = full_cols
                entry_offset += P.data.size
            gap_above += P.shape[0]
            acc_height += P_height * np.int64(P.shape[1])

        return sp.coo_matrix((vals, (rows, cols)), shape=(acc_height, num_params))

    def merge_q_list(
        self,
        q_list: List[sp.spmatrix | np.ndarray],
        constant: sp.csc_matrix,
        num_params: int,
    ) -> sp.csr_matrix:
        """Stack q with constant offset as last row.

        Args:
            q_list: list of q submatrices as scipy sparse matrices or numpy arrays.
            constant: The constant offset as a CSC sparse matrix.
            num_params: number of parameters in the problem.

        Returns:
            A CSR sparse representation of the merged q matrix.
        """
        # Fast path for no parameters.
        if num_params == 1:
            q = np.vstack(q_list)
            q = np.vstack([q, constant.A])
            return sp.csr_matrix(q)
        else:
            q = sp.vstack(q_list + [constant])
            return sp.csr_matrix(q)
