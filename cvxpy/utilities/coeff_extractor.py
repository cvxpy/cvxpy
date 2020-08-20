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

from __future__ import division

import operator

import numpy as np
import scipy.sparse as sp

from cvxpy.cvxcore.python import canonInterface
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (replace_quad_forms,
                                                restore_quad_forms)
from cvxpy.lin_ops.lin_op import LinOp, NO_OP


# TODO find best format for sparse matrices: csr, csc, dok, lil, ...
class CoeffExtractor(object):

    def __init__(self, inverse_data):
        self.id_map = inverse_data.var_offsets
        self.x_length = inverse_data.x_length
        self.var_shapes = inverse_data.var_shapes
        self.param_shapes = inverse_data.param_shapes
        self.param_to_size = inverse_data.param_to_size
        self.param_id_map = inverse_data.param_id_map

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
                                                 num_rows)

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
                                                         affine_expr.size)

        # TODO preserve sparsity
        # Iterates over every entry of the parameters vector,
        # and obtains the Pi and qi for that entry i.
        # These are then combined into matrices [P1.flatten(), P2.flatten(), ...]
        # and [q1, q2, ...]
        constant = param_coeffs[-1, :]
        c = param_coeffs[:-1, :].A

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
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                c_part = c[var_offset:var_offset+var_size, :]
                if quad_forms[var_id][2].P.value is not None:
                    # Convert to dense matrix.
                    P = quad_forms[var_id][2].P.value
                    if sp.issparse(P):
                        P = P.todense().A
                    if var_size == 1:
                        c_part = np.ones((P.shape[0], 1)) * c_part
                else:
                    P = np.eye(var_size)
                # We multiply the columns of P, by c_part.
                P = P[:, :, None] * c_part[None, :, :]
                if orig_id in coeffs:
                    if 'P' in coeffs[orig_id]:
                        coeffs[orig_id]['P'] += P
                    else:
                        coeffs[orig_id]['P'] = P
                else:
                    coeffs[orig_id] = dict()
                    coeffs[orig_id]['P'] = P
                    coeffs[orig_id]['q'] = np.zeros((P.shape[0], c.shape[1]))
            else:
                var_offset = affine_id_map[var.id][0]
                var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                if var.id in coeffs:
                    coeffs[var.id]['q'] += c[var_offset:var_offset+var_size, :]
                else:
                    coeffs[var.id] = dict()
                    coeffs[var.id]['q'] = c[var_offset:var_offset+var_size, :]
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
        for var_id, offset in offsets:
            shape = self.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            if var_id in coeffs and 'P' in coeffs[var_id]:
                P = coeffs[var_id]['P']
            else:
                P = np.zeros((size, size, num_params))
            if var_id in coeffs and 'q' in coeffs[var_id]:
                q = coeffs[var_id]['q']
            else:
                q = np.zeros((size, num_params))

            P_list.append(P)
            q_list.append(q)
            P_height += P.shape[0]

        if P_height != self.x_length:
            raise RuntimeError("Resulting quadratic form does not have "
                               "appropriate dimensions")

        # Conceptually we build a block diagonal matrix
        # out of all the Ps, then flatten the first two dimensions.
        # eg P1
        #      P2
        # We do this by extending each P with zero blocks above and below.
        gap_above = 0
        gap_below = P_height
        total_height = 0
        rows = []
        cols = []
        vals = []
        for P in P_list:
            gap_below -= P.shape[0]
            total_height += P_height * P.shape[1]
            # above = np.zeros((gap_above, P.shape[1], num_params))
            # below = np.zeros((gap_below, P.shape[1], num_params))
            # padded_P = np.concatenate([above, P, below], axis=0)
            # padded_P = np.reshape(padded_P, (P_height*P.shape[1], num_params),
            #                       order='F')
            # padded_P_list.append(padded_P)
            vals.extend(P.flatten(order='C'))
            base_rows = np.arange(gap_above, gap_above + P.shape[1])
            scale_rows = np.arange(P.shape[1]) * P_height
            P_rows = base_rows[None, :] + scale_rows[:, None]
            flat_rows = P_rows.flatten(order='C')
            rows.extend(np.repeat(flat_rows, num_params))
            cols.extend(np.repeat(np.arange(num_params), P.shape[0] * P.shape[1]))
            gap_above += P.shape[0]
            print(P)

        # Stitch together Ps and qs and constant.
        P = sp.coo_matrix((vals, (rows, cols)), shape=(total_height, num_params))
        print(P)
        # Stack q with constant offset as last row.
        q = np.vstack(q_list)
        q = np.vstack([q, constant.A])
        q = sp.csr_matrix(q)
        return P, q
