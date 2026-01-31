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
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
    ExpCone,
    OpRelEntrConeQuad,
    RelEntrConeQuad,
)
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
    von_neumann_entr_canon,
)
from cvxpy.utilities.power_tools import fracify, gm_constrs
from cvxpy.utilities.solver_context import SolverInfo

APPROX_CONE_CONVERSIONS = {
    RelEntrConeQuad: {SOC},
    OpRelEntrConeQuad: {cp.PSD},
    PowCone3D: {SOC},
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
        X = cp.reshape(X, (m, 1), order='F')
    #####################################
    # Comments from quad_over_lin_canon:
    #   quad_over_lin := sum_{i} x^2_{i} / y
    #   t = Variable(1,) is the epigraph variable.
    #   Becomes a constraint
    #   SOC(t=y + t, X=[y - t, 2*x])
    ####################################
    soc_X_col0 = cp.reshape(y - z, (m, 1), order='F')
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


def pow_3d_canon(con, args):
    """
    Convert PowCone3D to SOC constraints via rational approximation.

    con : PowCone3D
        The power cone constraint x^alpha * y^(1-alpha) >= |z|
    args : tuple of length three
        x, y, z = args[0], args[1], args[2]

    Returns a tuple (canon_constr, aux_constrs) where canon_constr is the first
    SOC constraint (used for id mapping) and aux_constrs are the remaining SOC constraints.
    """
    alpha = con.alpha
    x, y, z = args

    # Extract the numeric value from alpha (which may be a CVXPY expression)
    if hasattr(alpha, 'value'):
        alpha_val = alpha.value
    else:
        alpha_val = alpha

    # Convert alpha to numpy array for consistent handling
    alpha_arr = np.atleast_1d(np.asarray(alpha_val, dtype=float).flatten())

    # Handle scalar vs vector alpha
    if alpha_arr.size == 1:
        alpha_val = float(alpha_arr[0])
        # Convert alpha to rational approximation
        w, _ = fracify([alpha_val, 1 - alpha_val])

        # Flatten x, y, z if needed for element-wise constraints
        x_flat = reshape(x, (x.size,), order='F') if x.size > 1 else x
        y_flat = reshape(y, (y.size,), order='F') if y.size > 1 else y
        z_flat = reshape(z, (z.size,), order='F') if z.size > 1 else z

        # Create SOC constraints for each element
        all_constrs = []
        for i in range(max(x.size, 1)):
            xi = x_flat[i] if x.size > 1 else x_flat
            yi = y_flat[i] if y.size > 1 else y_flat
            zi = z_flat[i] if z.size > 1 else z_flat
            # gm_constrs creates: t <= x^w[0] * y^w[1]
            # We need: z <= x^alpha * y^(1-alpha)
            all_constrs.extend(gm_constrs(zi, [xi, yi], w))
    else:
        # Vector alpha - handle each element separately
        x_flat = reshape(x, (x.size,), order='F')
        y_flat = reshape(y, (y.size,), order='F')
        z_flat = reshape(z, (z.size,), order='F')

        all_constrs = []
        for i in range(alpha_arr.size):
            alpha_val = float(alpha_arr[i])
            w, _ = fracify([alpha_val, 1 - alpha_val])
            all_constrs.extend(gm_constrs(z_flat[i], [x_flat[i], y_flat[i]], w))

    # Return first constraint as canonical, rest as auxiliary
    # The Canonicalization class requires a non-None canon_constr for id mapping
    if all_constrs:
        return all_constrs[0], all_constrs[1:]
    else:
        # Edge case: no constraints generated (shouldn't happen in practice)
        raise ValueError("PowCone3D canonicalization produced no constraints")


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


def von_neumann_entr_canon_dispatch(expr, args, solver_context: SolverInfo | None = None):
    if expr.quad_approx:
        return von_neumann_entr_QuadApprox(expr, args)
    else:
        return von_neumann_entr_canon(expr, args)


class ApproxCone2Cone(Canonicalization):
    CANON_METHODS = {
        RelEntrConeQuad: RelEntrConeQuad_canon,
        OpRelEntrConeQuad: OpRelEntrConeQuad_canon,
        PowCone3D: pow_3d_canon,
    }

    def __init__(self, problem=None, target_cones=None) -> None:
        if target_cones is not None:
            canon_methods = {k: v for k, v in ApproxCone2Cone.CANON_METHODS.items()
                           if k in target_cones}
        else:
            canon_methods = dict(ApproxCone2Cone.CANON_METHODS)
        super(ApproxCone2Cone, self).__init__(
            problem=problem, canon_methods=canon_methods)
