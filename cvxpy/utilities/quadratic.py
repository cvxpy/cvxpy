"""
Copyright 2016 Jaehyun Park

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
import cvxpy as cvx
import numpy as np
import scipy.sparse as sp
import canonInterface
import cvxpy.lin_ops.lin_utils as lu
from numpy import linalg as LA

class QuadCoeffExtractor:
    def __init__(self, id_map, N):
        self.id_map = id_map
        self.N = N

    # TODO: determine the best sparse format for each of the
    #       quadratic atoms
    def _quad_coeffs_constant(self, expr):
        ret = [sp.lil_matrix((self.N+1, self.N+1)) for i in range(expr.size[0]*expr.size[1])]
        if expr.is_scalar():
            ret[0][self.N, self.N] = expr.value
        else:
            row = 0
            for j in range(expr.size[1]):
                for i in range(expr.size[0]):
                    ret[row][self.N, self.N] = expr.value[i, j]
                    row += 1
        return ret

    def _quad_coeffs_affine(self, expr):
        s, _ = expr.canonical_form
        V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(s)], self.id_map)
        ret = [sp.lil_matrix((self.N+1, self.N+1)) for i in range(expr.size[0]*expr.size[1])]
        for (v, i, j) in zip(V, I, J):
            ret[int(i)][j, self.N] = v
        for i, v in enumerate(b):
            ret[i][self.N, self.N] += v
        return ret

    def _quad_coeffs_affine_prod(self, expr):
        Xs = self.get_coeffs(expr.args[0])
        Ys = self.get_coeffs(expr.args[1])
        ret = []
        m, p = expr.args[0].size
        n = expr.args[1].size[1]
        for j in range(n):
            for i in range(m):
                M = sp.csc_matrix((self.N+1, self.N+1))
                for k in range(p):
                    Xind = k*m + i
                    Yind = j*p + k
                    M += Xs[Xind] * Ys[Yind].T
                ret.append(M)
        return ret

    # There might be a faster way
    def _quad_coeffs_quad_over_lin(self, expr):
        (A, b) = self.get_affine_coeffs(expr.args[0])
        P = A.T*A
        q = 2*b.T*A
        r = np.dot(b.T, b)
        y = expr.args[1].value
        return [sp.bmat([[P, None], [q, r]]) / y]

    def _quad_coeffs_power(self, expr):
        if expr.p == 1:
            return self.get_coeffs(expr.args[0])
        elif expr.p == 2:
            # (a^T x + b)^2 = x^T (a a^T) x + (2ba)^T x + b^2
            Xs = self.get_coeffs(expr.args[0])
            return [X*X.T for X in Xs]
        else:
            raise Exception("Error while processing power(x, %f)." % p)

    def _quad_coeffs_matrix_frac(self, expr):
        Xs = self.quad_coeffs(expr.args[0])
        Pinv = LA.inv(expr.args[1].value)
        m, n = expr.args[0].size
        M = sp.lil_matrix((self.N+1, self.N+1))
        for i in range(m):
            for j in range(m):
                M += sum([Pinv[i, j]*Xs[i+k*m]*Xs[j+k*m].T for k in range(n)])
        return [M]

    def _quad_coeffs_affine_atom(self, expr):
        Xs = []
        fake_args = []
        offsets = {}
        offset = 0
        for idx, arg in enumerate(expr.args):
            if arg.is_constant():
                fake_args += [lu.create_const(arg.value, arg.size)]
            else:
                Xs += self.quad_coeffs(arg)
                fake_args += [lu.create_var(arg.size, idx)]
                offsets[idx] = offset
                offset += arg.size[0]*arg.size[1]
        fake_expr, _ = expr.graph_implementation(fake_args, expr.size, expr.get_data())
        # Get the matrix representation of the function.
        V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(fake_expr)], offsets)
        # return "AX+b"
        ret = [sp.lil_matrix((self.N+1, self.N+1)) for i in range(expr.size[0]*expr.size[1])]
        for (v, i, j) in zip(V, I, J):
            ret[int(i)] += v*Xs[int(j)]
        for i, v in enumerate(b):
            ret[i][self.N, self.N] += v
        return ret

    def get_coeffs(self, expr):
        if expr.is_constant():
            return self._quad_coeffs_constant(expr)
        elif expr.is_affine():
            return self._quad_coeffs_affine(expr)
        elif isinstance(expr, cvx.affine_prod):
            return self._quad_coeffs_affine_prod(expr)
        elif isinstance(expr, cvx.quad_over_lin):
            return self._quad_coeffs_quad_over_lin(expr)
        elif isinstance(expr, cvx.power):
            return self._quad_coeffs_power(expr)
        elif isinstance(expr, cvx.matrix_frac):
            return self._quad_coeffs_matrix_frac(expr)
        elif isinstance(expr, cvx.affine.affine_atom.AffAtom):
            return self._quad_coeffs_affine_atom(expr)
        else:
            raise Exception("Unknown expression type %s." % type(expr))

    def get_affine_coeffs(self, expr):
        s, _ = expr.canonical_form
        V, I, J, b = canonInterface.get_problem_matrix([lu.create_eq(s)], self.id_map)
        A = sp.csr_matrix((V, (I, J)), shape=(expr.size[0]*expr.size[1], self.N))
        return (A, b)
