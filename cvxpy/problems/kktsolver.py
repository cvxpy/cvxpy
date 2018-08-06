"""
Copyright 2013 Steven Diamond

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

"""
THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""

# A custom KKT solver for CVXOPT that can handle redundant constraints.
# Uses regularization and iterative refinement.

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


def kkt_ldl(G, dims, A, mnl=0):
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
    from cvxopt import blas, lapack
    from cvxopt.base import matrix
    from cvxopt.misc import scale, pack, unpack

    p, n = A.size
    ldK = n + p + mnl + dims['l'] + sum(dims['q']) + sum([int(k*(k+1)/2)
                                                          for k in dims['s']])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))

    def factor(W, H=None, Df=None):
        blas.scal(0.0, K)
        if H is not None:
            K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            if mnl:
                g[:mnl] = Df[:, k]
            g[mnl:] = G[:, k]
            scale(g, W, trans='T', inverse='I')
            pack(g, K, dims, mnl, offsety=k*ldK + n + p)
        K[(ldK+1)*(p+n):: ldK+1] = -1.0
        # Add positive regularization in 1x1 block and negative in 2x2 block.
        K[0: (ldK+1)*n: ldK+1] += REG_EPS
        K[(ldK+1)*n:: ldK+1] += -REG_EPS
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
            blas.copy(y, u, offsety=n)
            scale(z, W, trans='T', inverse='I')
            pack(z, u, dims, mnl, offsety=n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n=n)
            blas.copy(u, y, offsetx=n, n=p)
            unpack(u, z, dims, mnl, offsetx=n + p)

        return solve

    return factor
