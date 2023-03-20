"""
Copyright 2022 the CVXPY developers

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

import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
    ExpCone,
    OpRelEntrConeQuad,
    RelEntrConeQuad,
)
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
    von_neumann_entr_canon,
)

APPROX_CONES = {
    RelEntrConeQuad: {cp.SOC},
    OpRelEntrConeQuad: {cp.PSD}
}


def gauss_legendre(n):
    """
    Helper function for returning the weights and nodes for an
    n-point Gauss-Legendre quadrature on [0, 1]
    """
    beta = 0.5/np.sqrt(np.ones(n-1)-(2*np.arange(1, n, dtype=float))**(-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D, V = np.linalg.eigh(T)
    x = D
    x, i = np.sort(x), np.argsort(x)
    w = 2 * (np.array([V[0][k] for k in i]))**2
    x = (x + 1)/2
    w = w/2
    return w, x


def rotated_quad_cone(X: cp.Expression, y: cp.Expression, z: cp.Expression):
    """
    For each i, enforce a constraint that
        (X[i, :], y[i], z[i])
    belongs to the rotated quadratic cone
        { (x, y, z) : || x ||^2 <= y z, 0 <= (y, z) }
    This implementation doesn't enforce (x, y) >= 0! That should be imposed by the calling function.
    """
    m = y.size
    assert z.size == m
    assert X.shape[0] == m
    if len(X.shape) < 2:
        X = cp.reshape(X, (m, 1))
    #####################################
    # Comments from quad_over_lin_canon:
    #   quad_over_lin := sum_{i} x^2_{i} / y
    #   t = Variable(1,) is the epigraph variable.
    #   Becomes a constraint
    #   SOC(t=y + t, X=[y - t, 2*x])
    ####################################
    soc_X_col0 = cp.reshape(y - z, (m, 1))
    soc_X = cp.hstack((soc_X_col0, 2*X))
    soc_t = y + z
    con = cp.SOC(t=soc_t, X=soc_X, axis=1)
    return con


def RelEntrConeQuad_canon(con: RelEntrConeQuad, args) -> Tuple[Constraint, List[Constraint]]:
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
    n = x.size
    # Z has been declared as so to allow for proper vectorization
    Z = Variable(shape=(k+1, n))
    w, t = gauss_legendre(m)
    T = Variable(shape=(m, n))
    lead_con = Zero(w @ T + con.z/2**k)
    constrs = [Zero(Z[0] - y)]

    for i in range(k):
        # The following matrix needs to be PSD.
        #     [Z[i]  , Z[i+1]]
        #     [Z[i+1], x     ]
        # The below recipe for imposing a 2x2 matrix as PSD follows from Pg-35, Ex 2.6
        # of Boyd's convex optimization. Where the constraint simply becomes a
        # rotated quadratic cone, see `dcp2cone/quad_over_lin_canon.py` for the very similar
        # scalar case
        epi = Z[i, :]
        stackedZ = Z[i+1, :]
        cons = rotated_quad_cone(stackedZ, epi, x)
        constrs.append(cons)
        constrs.extend([epi >= 0, x >= 0])

    for i in range(m):
        off_diag = -(t[i]**0.5) * T[i, :]
        # The following matrix needs to be PSD.
        #     [ Z[k] - x - T[i] , off_diag      ]
        #     [ off_diag        , x - t[i]*T[i] ]
        epi = (Z[k, :] - x - T[i, :])
        cons = rotated_quad_cone(off_diag, epi, x-t[i]*T[i, :])
        constrs.append(cons)
        constrs.extend([epi >= 0, x-t[i]*T[i, :] >= 0])

    return lead_con, constrs


def OpRelEntrConeQuad_canon(con: OpRelEntrConeQuad, args) -> Tuple[Constraint, List[Constraint]]:
    k, m = con.k, con.m
    X, Y = con.X, con.Y
    assert X.is_real()
    assert Y.is_real()
    assert con.Z.is_real()
    Zs = {i: Variable(shape=X.shape, symmetric=True) for i in range(k+1)}
    Ts = {i: Variable(shape=X.shape, symmetric=True) for i in range(m+1)}
    constrs = [Zero(Zs[0] - Y)]
    if not X.is_symmetric():
        ut = upper_tri(X)
        lt = upper_tri(X.T)
        constrs.append(ut == lt)
    if not Y.is_symmetric():
        ut = upper_tri(Y)
        lt = upper_tri(Y.T)
        constrs.append(ut == lt)
    if not con.Z.is_symmetric():
        ut = upper_tri(con.Z)
        lt = upper_tri(con.Z.T)
        constrs.append(ut == lt)
    w, t = gauss_legendre(m)
    lead_con = Zero(cp.sum([w[i] * Ts[i] for i in range(m)]) + con.Z/2**k)

    for i in range(k):
        #     [Z[i]  , Z[i+1]]
        #     [Z[i+1], x     ]
        constrs.append(cp.bmat([[Zs[i], Zs[i+1]], [Zs[i+1].T, X]]) >> 0)

    for i in range(m):
        off_diag = -(t[i]**0.5) * Ts[i]
        # The following matrix needs to be PSD.
        #     [ Z[k] - x - T[i] , off_diag      ]
        #     [ off_diag        , x - t[i]*T[i] ]
        constrs.append(cp.bmat([[Zs[k] - X - Ts[i], off_diag], [off_diag.T, X-t[i]*Ts[i]]]) >> 0)

    return lead_con, constrs


def von_neumann_entr_QuadApprox(expr, args):
    m, k = expr.quad_approx[0], expr.quad_approx[1]
    epi, initial_cons = von_neumann_entr_canon(expr, args)
    cons = []
    for con in initial_cons:
        if isinstance(con, ExpCone):  # should only hit this once.
            qa_con = con.as_quad_approx(m, k)
            qa_con_canon_lead, qa_con_canon = RelEntrConeQuad_canon(
                qa_con, None)
            cons.append(qa_con_canon_lead)
            cons.extend(qa_con_canon)
        else:
            cons.append(con)
    return epi, cons


def von_neumann_entr_canon_dispatch(expr, args):
    if expr.quad_approx:
        return von_neumann_entr_QuadApprox(expr, args)
    else:
        return von_neumann_entr_canon(expr, args)


class QuadApprox(Canonicalization):
    CANON_METHODS = {
        RelEntrConeQuad: RelEntrConeQuad_canon,
        OpRelEntrConeQuad: OpRelEntrConeQuad_canon
    }

    def __init__(self, problem=None) -> None:
        super(QuadApprox, self).__init__(
            problem=problem, canon_methods=QuadApprox.CANON_METHODS)
