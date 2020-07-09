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

        # TODO vectorize this code.
        # Iterates over every entry of the parameters vector,
        # and obtains the Pi and qi for that entry i.
        # These are then combined into matrices [P1.flatten(), P2.flatten(), ...]
        # and [q1, q2, ...]
        coeff_list = []
        constant = param_coeffs[-1, :]
        for p in range(param_coeffs.shape[1]):
            c = param_coeffs[:-1, p].A.flatten()

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
                    c_part = c[var_offset:var_offset+var_size]
                    if quad_forms[var_id][2].P.value is not None:
                        P = quad_forms[var_id][2].P.value
                        if c_part.size == 1:
                            P = c_part[0] * P
                        else:
                            P = P @ sp.diags(c_part)
                    else:
                        P = sp.diags(c_part)
                    if orig_id in coeffs:
                        coeffs[orig_id]['P'] += P
                    else:
                        coeffs[orig_id] = dict()
                        coeffs[orig_id]['P'] = P
                        coeffs[orig_id]['q'] = np.zeros(P.shape[0])
                else:
                    var_offset = affine_id_map[var.id][0]
                    var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                    if var.id in coeffs:
                        coeffs[var.id]['P'] += sp.csr_matrix((var_size, var_size))
                        coeffs[var.id]['q'] += c[var_offset:var_offset+var_size]
                    else:
                        coeffs[var.id] = dict()
                        coeffs[var.id]['P'] = sp.csr_matrix((var_size, var_size))
                        coeffs[var.id]['q'] = c[var_offset:var_offset+var_size]
            coeff_list.append(coeffs)
        return coeff_list, constant

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
        coeff_list, constant = self.extract_quadratic_coeffs(root.args[0],
                                                             quad_forms)
        # Restore expression.
        restore_quad_forms(root.args[0], quad_forms)

        # Sort variables corresponding to their starting indices, in ascending
        # order.
        offsets = sorted(self.id_map.items(), key=operator.itemgetter(1))

        # Get P and q for each parameter.
        P_list = []
        q_list = []
        for coeffs in coeff_list:
            # Concatenate quadratic matrices and vectors
            P = sp.csr_matrix((0, 0))
            q = np.zeros(0)
            for var_id, offset in offsets:
                if var_id in coeffs:
                    P = sp.block_diag([P, coeffs[var_id]['P']])
                    q = np.concatenate([q, coeffs[var_id]['q']])
                else:
                    shape = self.var_shapes[var_id]
                    size = np.prod(shape, dtype=int)
                    P = sp.block_diag([P, sp.csr_matrix((size, size))])
                    q = np.concatenate([q, np.zeros(size)])

            if (P.shape[0] != P.shape[1] and P.shape[1] != self.x_length) or \
                    q.shape[0] != self.x_length:
                raise RuntimeError("Resulting quadratic form does not have "
                                   "appropriate dimensions")
            if not np.isscalar(constant) and constant.size > 1:
                raise RuntimeError("Constant must be a scalar")

            P_size = P.shape[0]*P.shape[1]
            P_list.append(P.reshape((P_size, 1), order='F'))
            q_list.append(q)

        # Here we assemble the Ps and qs into matrices
        # that we multiply by a parameter vector to get P, q
        # i.e. [P1.flatten(), P2.flatten(), ...]
        #      [q1, q2, ...]
        # where Pi, qi are coefficients for the ith entry of
        # the parameter vector.

        # Stitch together Ps and qs and constant.
        P = sp.hstack(P_list)
        # Stack q with constant offset as last row.
        q = np.stack(q_list, axis=1)
        q = sp.vstack([q, constant])
        return P, q
