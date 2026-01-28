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

import numpy as np
from scipy import sparse

import cvxpy as cp
from cvxpy import problems
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution

EXACT_CONE_CONVERSIONS = {
    PowConeND: {PowCone3D},
    SOC: {PSD},
}
"""
Maps each cone to the set of cones it can be exactly converted to.
PowConeND -> PowCone3D via binary tree decomposition.
SOC -> PSD via Schur complement.
"""


def soc2psd_canon(con, args):
    """
    Convert a single SOC constraint ||X||_2 <= t to an equivalent PSD constraint.

    Uses the Schur complement formulation:
        [[t*I, X^T], [X, t*I]] >> 0

    For packed SOC constraints (vector t), produce multiple PSD constraints.
    Returns (canon_constr, aux_constrs) where canon_constr is the first PSD
    constraint and aux_constrs are the remaining PSD constraints.

    con : SOC
        The SOC constraint.
    args : tuple of length two
        t, X = args[0], args[1]
    """
    t, X = args

    if t.shape == (1,):
        # Single SOC constraint
        scalar_term = t[0]
        vector_term_len = X.shape[0]

        A = scalar_term * sparse.eye_array(1)
        B = reshape(X, (-1, 1), order='F').T
        C = scalar_term * sparse.eye_array(vector_term_len)

        M = cp.bmat([
            [A, B],
            [B.T, C]
        ])

        canon_constr = PSD(M)
        return canon_constr, []
    else:
        # Packed SOC constraints (vector t)
        if con.axis == 1:
            X = X.T
        psd_constraints = []
        for subidx in range(t.shape[0]):
            scalar_term = t[subidx]
            vector_term_len = X.shape[0]

            A = scalar_term * sparse.eye_array(1)
            B = X[:, subidx:subidx+1].T
            C = scalar_term * sparse.eye_array(vector_term_len)

            M = cp.bmat([
                [A, B],
                [B.T, C]
            ])

            psd_constraints.append(PSD(M))

        return psd_constraints[0], psd_constraints[1:]


def pow_nd_canon(con, args):
    """
    con : PowConeND
        We can extract metadata from this.
        For example, con.alpha and con.axis.
    args : tuple of length two
        W,z = args[0], args[1]
    """
    alpha, axis, _ = con.get_data()
    alpha = alpha.value
    W, z = args
    if axis == 1:
        W = W.T
        alpha = alpha.T
    if W.ndim == 1:
        W = reshape(W, (W.size, 1), order='F')
        alpha = np.reshape(alpha, (W.size, 1))
    n, k = W.shape
    if n == 2:
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        T = Variable(shape=(n-2, k))
        x_3d, y_3d, z_3d, alpha_3d = [], [], [], []
        for j in range(k):
            x_3d.append(W[:-1, j])
            y_3d.append(T[:, j])
            y_3d.append(W[n-1, j])
            z_3d.append(z[j])
            z_3d.append(T[:, j])
            r_nums = alpha[:, j]
            r_dens = np.cumsum(r_nums[::-1])[::-1]
            # ^ equivalent to [np.sum(alpha[i:, j]) for i in range(n)]
            r = r_nums / r_dens
            alpha_3d.append(r[:n-1])
        x_3d = hstack(x_3d)
        y_3d = hstack(y_3d)
        z_3d = hstack(z_3d)
        alpha_p3d = hstack(alpha_3d)
        # TODO: Ideally we should construct x,y,z,alpha_p3d by
        #   applying suitable sparse matrices to W,z,T, rather
        #   than using the hstack atom. (hstack will probably
        #   result in longer compile times).
        can_con = PowCone3D(x_3d, y_3d, z_3d, alpha_p3d)
    # Return a single PowCone3D constraint defined over all auxiliary
    # variables needed for the reduction to go through.
    # There are no "auxiliary constraints" beyond this one.
    return can_con, []


class ExactCone2Cone(Canonicalization):

    CANON_METHODS = {
        PowConeND: pow_nd_canon,
        SOC: soc2psd_canon,
    }

    def __init__(self, problem=None, target_cones=None) -> None:
        if target_cones is not None:
            canon_methods = {k: v for k, v in ExactCone2Cone.CANON_METHODS.items()
                           if k in target_cones}
        else:
            canon_methods = dict(ExactCone2Cone.CANON_METHODS)
        super(ExactCone2Cone, self).__init__(
            problem=problem, canon_methods=canon_methods)

    def apply(self, problem):
        """Override apply to track auxiliary PSD constraints for SOC dual recovery."""
        inverse_data = InverseData(problem)

        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective)

        # Track auxiliary PSD constraint IDs for packed SOC dual recovery.
        # Maps: soc_constraint_id -> [aux_psd_constraint_ids]
        soc_packed_aux = {}

        for constraint in problem.constraints:
            canon_constr, aux_constr = self.canonicalize_tree(constraint)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

            # Track auxiliary PSD constraints for packed SOC
            if isinstance(constraint, SOC) and SOC in self.canon_methods:
                aux_psd_ids = [c.id for c in aux_constr if isinstance(c, PSD)]
                if aux_psd_ids:
                    soc_packed_aux[constraint.id] = aux_psd_ids

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        inverse_data.soc_packed_aux = soc_packed_aux
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}

        if dvars == {}:
            # NOTE: pre-maturely trigger return of the method in case the problem
            # is infeasible (otherwise will run into some opaque errors)
            return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

        soc_packed_aux = getattr(inverse_data, 'soc_packed_aux', {})

        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND) and cons_id in dvars:
                div_size = int(dvars[cons_id].shape[1] // cons.args[1].shape[0])
                dv_list = []
                for i in range(cons.args[1].shape[0]):
                    # Iterating over the vectorized constraints
                    dv_list.append([])
                    tmp_duals = dvars[cons_id][:, i * div_size: (i + 1) * div_size]
                    for j, col_dvars in enumerate(tmp_duals.T):
                        if j == len(tmp_duals.T) - 1:
                            dv_list[-1] += [col_dvars[0], col_dvars[1]]
                        else:
                            dv_list[-1].append(col_dvars[0])
                    dv_list[-1].append(tmp_duals.T[0][-1])
                dvars[cons_id] = np.array(dv_list)

            elif isinstance(cons, SOC) and cons_id in dvars:
                # SOC dual recovery from PSD dual variables.
                # The first row of each PSD dual variable is the SOC dual variable,
                # scaled by 2.
                if cons_id in soc_packed_aux:
                    # Packed SOC: collect duals from canonical + auxiliary PSD constraints
                    parts = []
                    # First PSD constraint (canonical)
                    psd_dual = dvars[cons_id]
                    parts.append(psd_dual[0])
                    # Auxiliary PSD constraints
                    for aux_id in soc_packed_aux[cons_id]:
                        if aux_id in solution.dual_vars:
                            parts.append(solution.dual_vars[aux_id][0])
                    dvars[cons_id] = 2 * np.hstack(parts)
                else:
                    # Single SOC constraint: extract first row and scale by 2
                    dvars[cons_id] = 2 * dvars[cons_id][0]

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
