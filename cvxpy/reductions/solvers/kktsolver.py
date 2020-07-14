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

# A custom KKT solver for CVXOPT that can handle redundant constraints.
# Uses regularization and iterative refinement.

# Regularization constant.
REG_EPS = 1e-9


def setup_ldl_factor(c, G, h, dims, A, b):
    """
    The meanings of arguments in this function are identical to those of the
    function cvxopt.solvers.conelp. Refer to CVXOPT documentation

        https://cvxopt.org/userguide/coneprog.html#linear-cone-programs

    for more information.

    Note: CVXOPT allows G and A to be passed as dense matrix objects. However,
    this function will only ever be called with spmatrix objects. If creating
    a custom kktsolver of your own, you need to conform to this sparse matrix
    assumption.
    """
    factor = kkt_ldl(G, dims, A)
    return factor


def kkt_ldl(G, dims, A):
    """
    Returns a function handle "factor", which conforms to the CVXOPT
    custom KKT solver specifications:

        https://cvxopt.org/userguide/coneprog.html#exploiting-structure.

    For convenience, we provide a short outline for how this function works.

    First, we allocate workspace for use in "factor". The factor function is
    called with data (H, W). Once called, the factor function computes an LDL
    factorization of the 3 x 3 system:

        [ H           A'   G'*W^{-1}  ]
        [ A           0    0          ].
        [ W^{-T}*G    0   -I          ]

    Once that LDL factorization is computed, "factor" constructs another
    inner function, called "solve". The solve function uses the newly
    constructed LDL factorization to compute solutions to linear systems of
    the form

        [ H     A'   G'    ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ G     0   -W'*W  ]   [ uz ]   [ bz ]

    The factor function concludes by returning a reference to the solve function.

    Notes: In the 3 x 3 system, H is n x n, A is p x n, and G is N x n, where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ). For cone
    programs, H is the zero matrix.
    """
    from cvxopt import blas, lapack
    from cvxopt.base import matrix
    from cvxopt.misc import scale, pack, unpack

    p, n = A.size
    ldK = n + p + dims['l'] + sum(dims['q']) + sum([int(k*(k+1)/2)
                                                    for k in dims['s']])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (G.size[0], 1))

    def factor(W, H=None):
        blas.scal(0.0, K)
        if H is not None:
            K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            g[:] = G[:, k]
            scale(g, W, trans='T', inverse='I')
            pack(g, K, dims, 0, offsety=k*ldK + n + p)
        K[(ldK+1)*(p+n):: ldK+1] = -1.0
        # Add positive regularization in 1x1 block and negative in 2x2 block.
        K[0: (ldK+1)*n: ldK+1] += REG_EPS
        K[(ldK+1)*n:: ldK+1] += -REG_EPS
        lapack.sytrf(K, ipiv)

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'   G'*W^{-1}  ]   [ ux   ]   [ bx        ]
            #     [ A          0    0          ] * [ uy   [ = [ by        ]
            #     [ W^{-T}*G   0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.
            blas.copy(x, u)
            blas.copy(y, u, offsety=n)
            scale(z, W, trans='T', inverse='I')
            pack(z, u, dims, 0, offsety=n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n=n)
            blas.copy(u, y, offsetx=n, n=p)
            unpack(u, z, dims, 0, offsetx=n + p)

        return solve

    return factor
