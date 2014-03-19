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

from cvxopt import blas, lapack
from cvxopt.base import matrix
from cvxopt.misc import scale, pack, unpack

# Regularization constant.
REG_EPS = 1e-9
# Number of iterative refinement steps
ITER_REF = 3


# Returns a kktsolver for linear cone programs (or nonlinear if F is given).
def get_kktsolver(G, dims, A, F=None, linsolver='ldl'):
    if F is None:
        factor = kkt_types[linsolver](G, dims, A)
        def kktsolver(W):
            return factor(W)
    else:
        mnl, x0 = F()
        factor = kkt_types[linsolver](G, dims, A, mnl)
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
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        # Add positive regularization in 1x1 block and negative in 2x2 block.
        K[0 : (ldK+1)*n : ldK+1]  += REG_EPS
        K[(ldK+1)*n :: ldK+1]  += -REG_EPS
        lapack.sytrf(K, ipiv)

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
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            unpack(u, z, dims, mnl, offsetx = n + p)

        return solve

    return factor


def kkt_ldl_iter(G, dims, A, mnl = 0):
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
    Kcopy = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    rhs = matrix(0.0, (ldK,1))
    tmp = matrix(0.0, (ldK,1))
    g = matrix(0.0, (mnl + G.size[0], 1))

    def factor(W, H = None, Df = None):
        blas.scal(0.0, K)
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in xrange(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        blas.copy(K, Kcopy)
        # Add positive regularization in 1x1 block and negative in 2x2 block.
        K[0 : (ldK+1)*n : ldK+1]  += REG_EPS
        K[(ldK+1)*n :: ldK+1]  += -REG_EPS
        lapack.sytrf(K, ipiv)

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
            blas.copy(x, rhs)
            blas.copy(y, rhs, offsety = n)
            scale(z, W, trans = 'T', inverse = 'I')
            pack(z, rhs, dims, mnl, offsety = n + p)
            # At this stage, we want to solve K * x == rhs

            # we copy into u
            blas.copy(rhs, u)

            # solve once to get x0
            lapack.sytrs(K, ipiv, u)

            # Here, u contains the solution
            for _ in xrange(ITER_REF):
                # d = rhs - Kcopy * u
                blas.symv(Kcopy,u,tmp,alpha=-1.0)
                blas.axpy(rhs,tmp)
                #print blas.nrm2(tmp)
                if blas.nrm2(tmp) <= 1e-12:
                    #print "breaking at", _
                    break
                # solve A*d = tmp
                lapack.sytrs(K, ipiv, tmp)
                # u = u + tmp
                blas.axpy(tmp, u)

            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            unpack(u, z, dims, mnl, offsetx = n + p)

        return solve

    return factor

# KKT solvers
kkt_types = {
    'ldl': kkt_ldl,         # just adds static regularization
    'ldl_ir': kkt_ldl_iter  # uses static regularization with iterative refinement
}