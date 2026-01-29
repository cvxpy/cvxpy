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


class PowNDConversion:
    """PowConeND -> PowCone3D via binary tree decomposition."""
    source = PowConeND
    targets = {PowCone3D}

    @staticmethod
    def canonicalize(con, args):
        """
        con : PowConeND
            We can extract metadata from this.
            For example, con.alpha and con.axis.
        args : tuple of length two
            W,z = args[0], args[1]
        """
        alpha, axis, _, _ = con.get_data()
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

    @staticmethod
    def recover_dual(cons, dual_var, inverse_data, solution):
        div_size = int(dual_var.shape[1] // cons.args[1].shape[0])
        dv_list = []
        for i in range(cons.args[1].shape[0]):
            dv_list.append([])
            tmp_duals = dual_var[:, i * div_size: (i + 1) * div_size]
            for j, col_dvars in enumerate(tmp_duals.T):
                if j == len(tmp_duals.T) - 1:
                    dv_list[-1] += [col_dvars[0], col_dvars[1]]
                else:
                    dv_list[-1].append(col_dvars[0])
            dv_list[-1].append(tmp_duals.T[0][-1])
        return np.array(dv_list)


class SOCConversion:
    """SOC -> PSD via Schur complement."""
    source = SOC
    targets = {PSD}

    @staticmethod
    def canonicalize(con, args):
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

    @staticmethod
    def apply_hook(constraint, canon_constr, aux_constr, inverse_data):
        """Track auxiliary PSD constraint IDs for packed SOC dual recovery."""
        aux_psd_ids = [c.id for c in aux_constr if isinstance(c, PSD)]
        if aux_psd_ids:
            if not hasattr(inverse_data, 'soc_packed_aux'):
                inverse_data.soc_packed_aux = {}
            inverse_data.soc_packed_aux[constraint.id] = aux_psd_ids

    @staticmethod
    def recover_dual(cons, dual_var, inverse_data, solution):
        soc_packed_aux = getattr(inverse_data, 'soc_packed_aux', {})
        if cons.id in soc_packed_aux:
            parts = [dual_var[0]]
            for aux_id in soc_packed_aux[cons.id]:
                if aux_id in solution.dual_vars:
                    parts.append(solution.dual_vars[aux_id][0])
            return 2 * np.hstack(parts)
        else:
            return 2 * dual_var[0]


class ExactCone2Cone(Canonicalization):

    CONVERSIONS = [PowNDConversion, SOCConversion]

    CANON_METHODS = {c.source: c.canonicalize for c in CONVERSIONS}

    def __init__(self, problem=None, target_cones=None) -> None:
        conversions = self.CONVERSIONS
        if target_cones is not None:
            conversions = [c for c in conversions if c.source in target_cones]
        canon_methods = {c.source: c.canonicalize for c in conversions}
        self._dual_recovery = {c.source: c.recover_dual
                               for c in conversions if hasattr(c, 'recover_dual')}
        self._apply_hooks = {c.source: c.apply_hook
                             for c in conversions if hasattr(c, 'apply_hook')}
        super(ExactCone2Cone, self).__init__(
            problem=problem, canon_methods=canon_methods)

    def apply(self, problem):
        inverse_data = InverseData(problem)

        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective)

        for constraint in problem.constraints:
            canon_constr, aux_constr = self.canonicalize_tree(constraint)
            canon_constraints += aux_constr + [canon_constr]
            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

            hook = self._apply_hooks.get(type(constraint))
            if hook is not None:
                hook(constraint, canon_constr, aux_constr, inverse_data)

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}

        if not dvars:
            return Solution(solution.status, solution.opt_val, pvars, dvars,
                            solution.attr)

        for cons_id, cons in inverse_data.id2cons.items():
            recover = self._dual_recovery.get(type(cons))
            if recover is not None and cons_id in dvars:
                dvars[cons_id] = recover(cons, dvars[cons_id], inverse_data,
                                         solution)

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)


# EXACT_CONE_CONVERSIONS must be a DAG (no cycles allowed).
EXACT_CONE_CONVERSIONS = {c.source: c.targets for c in ExactCone2Cone.CONVERSIONS}
