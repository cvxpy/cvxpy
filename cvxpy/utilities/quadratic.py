"""
Copyright 2016 Jaehyun Park

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
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import canonInterface
import cvxpy.lin_ops.lin_utils as lu
from numpy import linalg as LA


class QuadCoeffExtractor(object):

    def __init__(self, id_map, N):
        self.id_map = id_map
        self.N = N

    # Given a quadratic expression expr of size m*n, extracts
    # the coefficients. Returns (Ps, Q, R) such that the (i, j)
    # entry of expr is given by
    #   x.T*Ps[k]*x + Q[k, :]*x + R[k],
    # where k = i + j*m. x is the vectorized variables indexed
    # by id_map.
    #
    # Ps: array of SciPy sparse matrices
    # Q: SciPy sparse matrix
    # R: NumPy array
    def get_coeffs(self, expr):
        if expr.is_constant():
            return self._coeffs_constant(expr)
        elif expr.is_affine():
            return self._coeffs_affine(expr)
        elif isinstance(expr, cvx.affine_prod):
            return self._coeffs_affine_prod(expr)
        elif isinstance(expr, cvx.quad_over_lin):
            return self._coeffs_quad_over_lin(expr)
        elif isinstance(expr, cvx.power):
            return self._coeffs_power(expr)
        elif isinstance(expr, cvx.matrix_frac):
            return self._coeffs_matrix_frac(expr)
        elif isinstance(expr, cvx.affine.affine_atom.AffAtom):
            return self._coeffs_affine_atom(expr)
        else:
            raise Exception("Unknown expression type %s." % type(expr))

    # TODO: determine the best sparse format for each of the
    #       quadratic atoms
    def _coeffs_constant(self, expr):
        if expr.is_scalar():
            sz = 1
            R = np.array([expr.value])
        else:
            sz = expr.size[0]*expr.size[1]
            R = expr.value.reshape(sz, order='F')
        Ps = [sp.csr_matrix((self.N, self.N)) for i in range(sz)]
        Q = sp.csr_matrix((sz, self.N))
        return (Ps, Q, R)

    def _coeffs_affine(self, expr):
        sz = expr.size[0]*expr.size[1]
        s, _ = expr.canonical_form
        V, I, J, R = canonInterface.get_problem_matrix([lu.create_eq(s)], self.id_map)
        Q = sp.csr_matrix((V, (I, J)), shape=(sz, self.N))
        Ps = [sp.csr_matrix((self.N, self.N)) for i in range(sz)]
        return (Ps, Q, R.flatten())

    def _coeffs_affine_prod(self, expr):
        (_, XQ, XR) = self._coeffs_affine(expr.args[0])
        (_, YQ, YR) = self._coeffs_affine(expr.args[1])

        m, p = expr.args[0].size
        n = expr.args[1].size[1]

        Ps = []
        Q = sp.csr_matrix((m*n, self.N))
        R = np.zeros((m*n))

        ind = 0
        for j in range(n):
            for i in range(m):
                M = sp.csr_matrix((self.N, self.N))  # TODO: find best format
                for k in range(p):
                    Xind = k*m + i
                    Yind = j*p + k

                    a = XQ[Xind, :]
                    b = XR[Xind]
                    c = YQ[Yind, :]
                    d = YR[Yind]

                    M += a.T*c
                    Q[ind, :] += b*c + d*a
                    R[ind] += b*d

                Ps.append(M.tocsr())
                ind += 1

        return (Ps, Q.tocsr(), R)

    def _coeffs_quad_over_lin(self, expr):
        (_, A, b) = self._coeffs_affine(expr.args[0])
        P = A.T*A
        q = sp.csr_matrix(2*b.T*A)
        r = np.dot(b.T, b)
        y = expr.args[1].value
        return ([P/y], q/y, np.array([r/y]))

    def _coeffs_power(self, expr):
        if expr.p == 1:
            return self.get_coeffs(expr.args[0])
        elif expr.p == 2:
            (_, A, b) = self._coeffs_affine(expr.args[0])
            Ps = [(A[i, :].T*A[i, :]).tocsr() for i in range(A.shape[0])]
            Q = 2*(sp.diags(b, 0)*A).tocsr()
            R = np.power(b, 2)
            return (Ps, Q, R)
        else:
            raise Exception("Error while processing power(x, %f)." % expr.p)

    def _coeffs_matrix_frac(self, expr):
        (_, A, b) = self._coeffs_affine(expr.args[0])
        m, n = expr.args[0].size
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

        return ([M.tocsr()], Q.tocsr(), np.array([R]))

    def _coeffs_affine_atom(self, expr):
        sz = expr.size[0]*expr.size[1]
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
                fake_args += [lu.create_const(arg.value, arg.size)]
            else:
                if Parg is None:
                    (Parg, Qarg, Rarg) = self.get_coeffs(arg)
                else:
                    (p, q, r) = self.get_coeffs(arg)
                    Parg += p
                    Qarg = sp.vstack([Qarg, q])
                    Rarg = np.concatenate([Rarg, r])
                fake_args += [lu.create_var(arg.size, idx)]
                offsets[idx] = offset
                offset += arg.size[0]*arg.size[1]
        fake_expr, _ = expr.graph_implementation(fake_args, expr.size, expr.get_data())
        # Get the matrix representation of the function.
        V, I, J, R = canonInterface.get_problem_matrix([lu.create_eq(fake_expr)], offsets)
        R = R.flatten()
        # return "AX+b"
        for (v, i, j) in zip(V, I.astype(int), J.astype(int)):
            Ps[i] += v*Parg[j]
            Q[i, :] += v*Qarg[j, :]
            R[i] += v*Rarg[j]

        Ps = [P.tocsr() for P in Ps]
        return (Ps, Q.tocsr(), R)
