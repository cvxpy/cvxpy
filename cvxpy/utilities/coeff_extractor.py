"""
Copyright 2016 Jaehyun Park, 2017 Robin Verschueren

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

from __future__ import division

import operator

import canonInterface
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA

import cvxpy as cvx
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.qp2quad_form.replace_quad_forms import ReplaceQuadForms


# TODO find best format for sparse matrices: csr, csc, dok, lil, ...
class CoeffExtractor(object):

    def __init__(self, inverse_data):
        self.id_map = inverse_data.var_offsets
        self.N = inverse_data.x_length
        self.var_shapes = inverse_data.var_shapes

    # Given a quadratic expression expr of shape m*n, extracts
    # the coefficients. Returns (Ps, Q, R) such that the (i, j)
    # entry of expr is given by
    #   x.T*Ps[k]*x + Q[k, :]*x + R[k],
    # where k
    # V= i + j*m. x is the vectorized variables indexed
    # by id_map.
    #
    # Ps: array of SciPy sparse matrices
    # Q: SciPy sparse matrix
    # R: NumPy array
    def get_coeffs(self, expr):
        if expr.is_constant():
            return self.constant(expr)
        elif expr.is_affine():
            return self.affine(expr)
        elif isinstance(expr, cvx.affine_prod):
            return self.affine_prod(expr)
        elif isinstance(expr, cvx.quad_over_lin):
            return self.quad_over_lin(expr)
        elif isinstance(expr, cvx.power):
            return self.power(expr)
        elif isinstance(expr, cvx.matrix_frac):
            return self.matrix_frac(expr)
        elif isinstance(expr, cvx.affine.affine_atom.AffAtom):
            return self.affine_atom(expr)
        elif expr.is_quadratic():
            return self.quad_form(expr)
        else:
            raise Exception("Unknown expression type %s." % type(expr))

    def constant(self, expr):
        size = expr.shape[0]*expr.shape[1]
        return sp.csr_matrix((size, self.N)), expr.value.reshape(size, order='F')

    def affine(self, expr):
        """ If expression is A*x + b, return A, b
        """
        if not expr.is_affine():
            raise ValueError("Expression is not affine")
        size = expr.shape[0]*expr.shape[1]
        s, _ = expr.canonical_form
        V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(s)], self.id_map)
        A = sp.csr_matrix((V, (I, J)), shape=(size, self.N))
        return A, b.flatten()

    def affine_prod(self, expr):
        XQ, XR = self.affine(expr.args[0])
        YQ, YR = self.affine(expr.args[1])

        m, p = expr.args[0].shape
        n = expr.args[1].shape[1]

        Ps = []
        Q = sp.csr_matrix((m*n, self.N))
        R = np.zeros((m*n))

        ind = 0
        for j in range(n):
            for i in range(m):
                M = sp.csr_matrix((self.N, self.N))
                for k in range(p):
                    Xind = k*m + i
                    Yind = j*p + k

                    a = XQ[Xind, :]
                    b = XR[Xind]
                    c = YQ[Yind, :]
                    d = YR[Yind]

                    M += a*c.T
                    Q[ind, :] += b*c + d*a
                    R[ind] += b*d

                Ps.append(M.tocsr())
                ind += 1

        return (Ps, Q.tocsr(), R)

    def quad_over_lin(self, expr):
        A, b = self.affine(expr.args[0])
        P = A.T*A
        q = sp.csr_matrix(2*b.T*A)
        r = np.dot(b.T, b)
        y = float(expr.args[1].value)
        return [P/y], q/y, np.array([r/y])

    def power(self, expr):
        if expr.p == 1:
            return self.get_coeffs(expr.args[0])
        elif expr.p == 2:
            A, b = self.affine(expr.args[0])
            Ps = [(A[i, :].T*A[i, :]).tocsr() for i in range(A.shape[0])]
            Q = 2*(sp.diags(b, 0)*A).tocsr()
            R = np.power(b, 2)
            return Ps, Q, R
        else:
            raise Exception("Error while processing power(x, %f)." % expr.p)

    def matrix_frac(self, expr):
        A, b = self.affine(expr.args[0])
        m, n = expr.args[0].shape
        Pinv = np.asarray(LA.inv(expr.args[1].value))

        M = sp.lil_matrix((self.N, self.N))
        Q = sp.lil_matrix((1, self.N))
        R = 0

        for i in range(0, m*n, m):
            A2 = A[i:i+m, :]
            b2 = b[i:i+m]

            M += A2.T*Pinv*A2
            Q += 2*A2.T.dot(np.dot(Pinv, b2))
            R += np.dot(b2, np.dot(Pinv, b2))

        return [M.tocsr()], Q.tocsr(), np.array([R])

    def affine_atom(self, expr):
        sz = expr.shape[0]*expr.shape[1]
        Ps = [sp.lil_matrix((self.N, self.N)) for i in range(sz)]
        Q = sp.lil_matrix((sz, self.N))
        Parg = None
        Qarg = None
        Rarg = None

        fake_args = []
        offsets = {}
        offset = 0
        for idx, arg in enumerate(expr.args):
            if arg.is_constant():
                fake_args += [lu.create_const(arg.value, arg.shape)]
            else:
                if Parg is None:
                    (Parg, Qarg, Rarg) = self.get_coeffs(arg)
                else:
                    (p, q, r) = self.get_coeffs(arg)
                    Parg += p
                    Qarg = sp.vstack([Qarg, q])
                    Rarg = np.vstack([Rarg, r])
                fake_args += [lu.create_var(arg.shape, idx)]
                offsets[idx] = offset
                offset += arg.shape[0]*arg.shape[1]
        fake_expr, _ = expr.graph_implementation(fake_args, expr.shape, expr.get_data())
        # Get the matrix representation of the function.
        V, I, J, R = canonInterface.get_problem_matrix([lu.create_eq(fake_expr)], offsets)
        R = R.flatten()
        # return "AX+b"
        for (v, i, j) in zip(V, I.astype(int), J.astype(int)):
            Ps[i] += v*Parg[j]
            Q[i, :] += v*Qarg[j, :]
            R[i] += v*Rarg[j]

        Ps = [P.tocsr() for P in Ps]
        return Ps, Q.tocsr(), R

    def fill_in_quad_forms(self, affine_problem, quad_forms):
        affine_inverse_data = InverseData(affine_problem)
        affine_id_map = affine_inverse_data.id_map
        affine_var_shapes = affine_inverse_data.var_shapes
        extractor = CoeffExtractor(affine_inverse_data)
        c, b = extractor.affine(affine_problem.objective.expr)
        coeffs = {}
        for var in affine_problem.variables():
            if var.id in quad_forms:
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                if quad_forms[var_id][2].P is not None:
                    c_part = c[0, var_offset:var_offset+var_size].toarray().flatten()
                    P = c_part * quad_forms[var_id][2].P.value
                else:
                    P = sp.diags(c[0, var_offset:var_offset+var_size].toarray().flatten())
                coeffs[orig_id] = dict()
                coeffs[orig_id]['P'] = P
                coeffs[orig_id]['q'] = np.zeros(P.shape[0])
            else:
                var_offset = affine_id_map[var.id][0]
                var_shape = affine_var_shapes[var.id]
                n = var_shape[0]
                var_size = var_shape[0]*var_shape[1]
                coeffs[var.id] = dict()
                coeffs[var.id]['P'] = sp.csr_matrix((n, n))
                coeffs[var.id]['q'] = c[0, var_offset:var_offset+var_size].toarray().flatten()
        return coeffs, b

    def quad_form(self, problem):
        """ Extract quadratic, linear and constant part of a quadratic objective
        """
        affine_problem, quad_forms = ReplaceQuadForms().apply(problem)
        coeffs, constant = self.fill_in_quad_forms(affine_problem, quad_forms)
        ReplaceQuadForms().invert(affine_problem, quad_forms)
        P = sp.csr_matrix((0, 0))
        q = np.zeros(0)
        offsets = sorted(self.id_map.items(), key=operator.itemgetter(1))
        for var_id, offset in offsets:
            if var_id in coeffs:
                P = sp.block_diag([P, coeffs[var_id]['P']])
                q = np.concatenate([q, coeffs[var_id]['q']])
            else:
                shape = self.var_shapes[var_id]
                size = shape[0]*shape[1]
                P = sp.block_diag([P, sp.csr_matrix((size, size))])
                q = np.concatenate([q, np.zeros(size)])

        if P.shape[0] != P.shape[1] != self.N or q.shape[0] != self.N:
            raise RuntimeError("Resulting quadratic form does not have appropriate dimensions")
        return P.tocsr(), q, constant
