"""
Copyright 2013 Steven Diamond

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

# A custom KKT solver for CVXOPT that can handle redundant constraints.
# Uses regularization and iterative refinement.

from cvxopt import blas, lapack, cholmod
from cvxopt.base import matrix, spmatrix
from cvxopt.misc import scale, pack, unpack
import math

# Regularization constant.
REG_EPS = 1e-9

# Returns a kktsolver for linear cone programs (or nonlinear if F is given).
def get_kktsolver(G, dims, A, F=None):
    if F is None:
        factor = kkt_ldl(G, dims, A)
        def kktsolver(W):
            return factor(W)
    else:
        mnl, x0 = F()
        factor = kkt_ldl(G, dims, A, mnl)
        def kktsolver(x, z, W):
            f, Df, H = F(x, z)
            return factor(W, H, Df)
    return kktsolver

def kkt_ldl(G, dims, A, mnl = 0):
    """
    Solution of KKT equations by a dense LDL factorization of the
    3 x 3 system.

    Returns a function that (1) computes the LDL factorization of

        [ H           A'   GG'*W^{-1} ]
        [ A           0    0          ],
        [ W^{-T}*GG   0   -I          ]

    given H, Df, W, where GG = [Df; G], and (2) returns a function for
    solving

        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]

    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """

    p, n = A.size
    ldK = n + p + mnl + dims['l'] + sum(dims['q']) + sum([ int(k*(k+1)/2)
        for k in dims['s'] ])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))

    def factor(W, H = None, Df = None):
        blas.scal(0.0, K)
        # K = spmatrix(0.0, [], [], size=(ldK, ldK))
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            sparse_pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        # Add positive regularization in 1x1 block and negative in 2x2 block.
        for i in range(0, (ldK+1)*n, ldK+1):
            K[i] += REG_EPS
        for i in range((ldK+1)*n, ldK*ldK, ldK+1):
            K[i] -= REG_EPS
        lapack.sytrf(K, ipiv)
        # Factor K as LDL'.
        # cholmod.options['supernodal'] = 1
        # ipiv = cholmod.symbolic(K, uplo = 'L')
        # try:
        #     cholmod.numeric(K, ipiv)
        # except ArithmeticError, e:
        #     flag = True
        #     K = matrix(K)
        #     ipiv = matrix(0, (ldK, 1))
        #     lapack.sytrf(K, ipiv)
        #     print "hello"

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0    0          ] * [ uy   [ = [ by        ]
            #     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.
            blas.copy(x, u)
            blas.copy(y, u, offsety = n)
            scale(z, W, trans = 'T', inverse = 'I')
            pack(z, u, dims, mnl, offsety = n + p)
            # Backsolves using LDL' factorization of K.
            lapack.sytrs(K, ipiv, u)
            #cholmod.solve(ipiv, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            unpack(u, z, dims, mnl, offsetx = n + p)

        return solve

    return factor

def sparse_pack(x, y, dims, mnl = 0, offsetx = 0, offsety = 0):
    """
    Copy x to y using packed storage.

    The vector x is an element of S, with the 's' components stored in
    unpacked storage.  On return, x is copied to y with the 's' components
    stored in packed storage and the off-diagonal entries scaled by
    sqrt(2).
    """

    nlq = mnl + dims['l'] + sum(dims['q'])
    # Copies elements from x to y, starting from offsetx in x
    # and offsety in y.
    y[offsety:offsety+nlq] = x[offsetx:offsetx+nlq]
    #blas.copy(x, y, n = nlq, offsetx = offsetx, offsety = offsety)
    iu, ip = offsetx + nlq, offsety + nlq
    for n in dims['s']:
       for k in range(n):
           startx = iu + k*(n+1)
           starty = ip
           y[starty:starty + n-k] = x[startx:startx + n-k]
           #blas.copy(x, y, n = n-k, offsetx = iu + k*(n+1), offsety = ip)
           y[ip] /= math.sqrt(2)
           ip += n-k
       iu += n**2
    np = sum([ int(n*(n+1)/2) for n in dims['s'] ])
    offset = offsety+nlq
    y[offset:offset+np] *= math.sqrt(2.0)
    #blas.scal(math.sqrt(2.0), y, n = np, offset = offsety+nlq)
