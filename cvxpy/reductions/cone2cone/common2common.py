"""
Copyright 2021 the CVXPY developers

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

from typing import List, Tuple

import numpy as np

from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import ExpConeQuad
from cvxpy.constraints.nonpos import NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import (
    quad_over_lin_canon,)

COMMON_CONES = {
    ExpConeQuad: {quad_over_lin}
}

# def gauss_legendre(n) -> Tuple[np.array, np.array]:
#     """
#     Helper function for returning the weights and nodes for an
#     n+1-point Gauss-Legendre quadrature
#     """
#     I = fixed_quad(scipy.log, a=0, b=1, n=n)
#     beta = 0.5/np.sqrt(np.ones(n)-(2*np.arange(1, n+1, dtype=float))**(-2))
#     T = np.diag(beta, 1) + np.diag(beta, -1)
#     V, D = np.linalg.eig(T)
#     x = np.diag(D)
#     x, i = np.sort(x), np.argsort(x)
#     w = 2*np.array([V[k] for k in i])**2


def gauss_legendre(n) -> Tuple[np.array, np.array]:
    """
    Helper function for returning the weights and nodes for an
    n-point Gauss-Legendre quadrature on [0, 1]
    """
    beta = 0.5/np.sqrt(np.ones(n)-(2*np.arange(1, n+1, dtype=float))**(-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D, V = np.linalg.eigh(T)
    # print("shape", V.shape)
    s = np.diag(V)
    # s, i = np.sort(s), np.argsort(s)
    s = np.sort(s)   # Riley Q: why aren't we using "i"?
    # w = 2*D[0,i]**2
    w = 2*D**2
    # translate and scale to [0, 1]
    s = (s + 1)/2
    w = w/2
    return w, s


def ExpConeQuad_canon(con: ExpConeQuad, args) -> Tuple[Constraint, List[Constraint]]:
    """
    Use linear and SOC constraints to approximately enforce
        con.x * log(con.x / con.y) <= con.z.

    We rely on an SOC characterization of 2-by-2 PSD matrices.
    Namely, a matrix
        [ a, b ]
        [ b, c ]
    is PSD if and only if (a, c) >= 0 and a*c >= b**2.
    That system of constraints can be expressed as
        a >= quad_over_lin(b, c).

    Note: constraint canonicalization in CVXPY uses a return format
    (lead_con, con_list) where lead_con is a Constraint that might be
    used in dual variable recovery and con_list consists of extra
    Constraint objects as needed.
    """
    k, m = con.k, con.m
    x, y = con.x, con.y
    Z = Variable(shape=(k+1,))
    w, t = gauss_legendre(m-1)
    T = Variable(m)
    lead_con = Zero(w @ T + con.z / 2**k)
    constrs = [Zero(Z[0] - y)]

    for i in range(k):
        # The following matrix needs to be PSD.
        #     [Z[i]  , Z[i+1]]
        #     [Z[i+1], x     ]
        expr = quad_over_lin(Z[i+1], x)
        epi, cons = quad_over_lin_canon(expr, expr.args)
        constrs.extend(cons)
        constrs.append(NonPos(epi-Z[i]))

    for i in range(m):
        # print("see here:", t[i])
        off_diag = -(t[i]**0.5) * T[i]
        # The following matrix needs to be PSD.
        #     [ Z[k] - x - T[i] , off_diag      ]
        #     [ off_diag        , x - t[i]*T[i] ]
        expr = quad_over_lin(off_diag, x - t[i]*T[i])
        epi, cons = quad_over_lin_canon(expr, expr.args)
        constrs.extend(cons)
        constrs.append(NonPos(epi - (Z[k] - x - T[i])))

    return lead_con, constrs


class Common2Common(Canonicalization):

    CANON_METHODS = {
        ExpConeQuad: ExpConeQuad_canon
    }

    def __init__(self, problem=None) -> None:
        super(Common2Common, self).__init__(
            problem=problem, canon_methods=Common2Common.CANON_METHODS)
