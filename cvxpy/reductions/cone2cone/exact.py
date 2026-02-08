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

Exact cone-to-cone conversions that preserve problem equivalence.

Each conversion is a class with:
  - source / targets: the cone type converted and the set it maps to.
  - canonicalize(con, args): rewrite one constraint.
  - apply_hook(constraint, canon_constr, aux_constr, inverse_data)
        [optional]: store metadata on inverse_data after canonicalization,
        used later by recover_dual.
  - recover_dual(cons, dual_var, inverse_data, solution)
        [optional]: map solver duals back to the original cone's duals.

Current conversions:
  PowConeND -> PowCone3D  (balanced binary tree decomposition)
  SOC       -> PSD        (Schur complement)

EXACT_CONE_CONVERSIONS maps {source_cone: {target_cones}} and must
form a DAG (no cycles).
"""

import numpy as np
from scipy import sparse

import cvxpy as cp
from cvxpy import problems
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.complex_psd import ComplexPSD
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cone2cone.cone_tree import (
    LeafNode,
    SingleVarNode,
    SplitNode,
    get_root_cone_id,
)
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solution import Solution

# Maps each "exact" cone to the set of simpler cones it converts to.
# Must form a DAG (no cycles).
EXACT_CONE_CONVERSIONS = {
    PowConeND: {PowCone3D},
    SOC: {PSD},
    ComplexPSD: {PSD},
}


def _build_pow_tree(
    W, alpha, z, j, indices, alphas,
    x_3d, y_3d, z_3d, alpha_3d,
    is_root=False,
):
    """Recursively build a balanced binary tree for PowConeND decomposition."""
    n = len(indices)

    if n == 1:
        # Base case: single variable, no cone needed
        tree = SingleVarNode(var_index=indices[0]) if j == 0 else None
        return W[indices[0], j], alphas[0], tree

    if n == 2:
        # Base case: two variables, create one 3D cone
        i0, i1 = indices
        a0, a1 = alphas
        total = a0 + a1
        r = a0 / total

        if is_root:
            out_expr = z[j]
        else:
            out_expr = Variable(shape=())

        x_3d.append(W[i0, j])
        y_3d.append(W[i1, j])
        z_3d.append(out_expr)
        alpha_3d.append(r)

        tree = LeafNode(
            cone_id=len(x_3d) - 1,
            var_indices=(i0, i1)
        ) if j == 0 else None

        return out_expr, total, tree

    # Recursive case: split into two halves
    mid = n // 2
    left_indices = indices[:mid]
    right_indices = indices[mid:]
    left_alphas = alphas[:mid]
    right_alphas = alphas[mid:]

    left_expr, left_sum, left_tree = _build_pow_tree(
        W, alpha, z, j, left_indices, left_alphas,
        x_3d, y_3d, z_3d, alpha_3d, is_root=False
    )
    right_expr, right_sum, right_tree = _build_pow_tree(
        W, alpha, z, j, right_indices, right_alphas,
        x_3d, y_3d, z_3d, alpha_3d, is_root=False
    )

    total = left_sum + right_sum
    r = left_sum / total

    if is_root:
        out_expr = z[j]
    else:
        out_expr = Variable(shape=())

    x_3d.append(left_expr)
    y_3d.append(right_expr)
    z_3d.append(out_expr)
    alpha_3d.append(r)

    if j == 0:
        tree = SplitNode(
            cone_id=len(x_3d) - 1,
            left=left_tree,
            right=right_tree
        )
    else:
        tree = None

    return out_expr, total, tree


def _extract_pow_duals_recursive(
    node, parent_cone_id, is_left_child,
    raw_duals, w_duals, j, num_cones_per_col,
):
    """Recursively extract W duals from the tree.

    For each node:
    - SingleVarNode: the variable's dual comes from the parent cone's
      x (row 0) or y (row 1) position depending on whether this is
      the left or right child.
    - LeafNode: both variable duals come from this node's own cone
      (x in row 0, y in row 1).
    - SplitNode: recurse into children. The split node's cone combines
      the left and right subtree outputs.
    """
    if isinstance(node, SingleVarNode):
        assert parent_cone_id is not None
        col = j * num_cones_per_col + parent_cone_id
        row = 0 if is_left_child else 1
        w_duals[node.var_index, j] = raw_duals[row, col]
    elif isinstance(node, LeafNode):
        col = j * num_cones_per_col + node.cone_id
        w_duals[node.var_indices[0], j] = raw_duals[0, col]
        w_duals[node.var_indices[1], j] = raw_duals[1, col]
    elif isinstance(node, SplitNode):
        _extract_pow_duals_recursive(
            node.left, node.cone_id, True,
            raw_duals, w_duals, j, num_cones_per_col
        )
        _extract_pow_duals_recursive(
            node.right, node.cone_id, False,
            raw_duals, w_duals, j, num_cones_per_col
        )


def _extract_pow_duals(tree, raw_duals, n, k, num_cones_per_col):
    """Extract W and z duals from raw 3D cone duals using tree structure."""
    w_duals = np.zeros((n, k))
    z_duals = np.zeros(k)

    root_idx = get_root_cone_id(tree)

    for j in range(k):
        _extract_pow_duals_recursive(
            tree, None, True,
            raw_duals, w_duals, j, num_cones_per_col
        )

        if root_idx is not None:
            root_col = j * num_cones_per_col + root_idx
            z_duals[j] = raw_duals[2, root_col]

    return w_duals, z_duals


class PowNDConversion:
    """PowConeND -> PowCone3D via balanced binary tree decomposition.

    Consolidated from the former exotic2common.py. Uses O(log n) tree
    depth instead of a linear chain for improved numerical stability.
    """
    source = PowConeND
    targets = {PowCone3D}

    @staticmethod
    def canonicalize(con, args):
        """
        Canonicalize PowConeND to PowCone3D using balanced binary tree
        decomposition.

        con : PowConeND
            We can extract metadata from this.
            For example, con.alpha and con.axis.
        args : tuple of length two
            W, z = args[0], args[1]
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
            return can_con, []

        # Balanced tree decomposition for n > 2
        x_3d, y_3d, z_3d, alpha_3d = [], [], [], []
        tree = None

        for j in range(k):
            indices = list(range(n))
            alphas = list(alpha[:, j])

            _, _, col_tree = _build_pow_tree(
                W, alpha, z, j, indices, alphas,
                x_3d, y_3d, z_3d, alpha_3d, is_root=True
            )

            if j == 0:
                tree = col_tree

        x_3d_expr = hstack(x_3d)
        y_3d_expr = hstack(y_3d)
        z_3d_expr = hstack(z_3d)
        alpha_p3d = hstack(alpha_3d)
        # TODO: Ideally we should construct x,y,z,alpha_p3d by
        #   applying suitable sparse matrices to W,z,T, rather
        #   than using the hstack atom. (hstack will probably
        #   result in longer compile times).
        can_con = PowCone3D(x_3d_expr, y_3d_expr, z_3d_expr, alpha_p3d)

        # Store tree on the constraint for the apply_hook to retrieve.
        can_con._pow_tree = tree

        return can_con, []

    @staticmethod
    def apply_hook(constraint, canon_constr, aux_constr, inverse_data):
        """Store the binary tree structure for dual variable recovery.

        Called by ExactCone2Cone.apply() after canonicalizing each
        PowConeND constraint. The tree records the balanced binary
        decomposition so that recover_dual() can map PowCone3D duals
        back to PowConeND duals.
        """
        tree = getattr(canon_constr, '_pow_tree', None)
        if tree is not None:
            if not hasattr(inverse_data, 'pow_tree_mappings'):
                inverse_data.pow_tree_mappings = {}
            inverse_data.pow_tree_mappings[constraint.id] = tree

    @staticmethod
    def recover_dual(cons, dual_var, inverse_data, solution):
        """Recover PowConeND dual variables from PowCone3D duals."""
        alpha, axis, _ = cons.get_data()
        n = alpha.shape[0] if axis == 0 else alpha.shape[1]
        k = cons.args[1].shape[0]

        tree_mappings = getattr(inverse_data, 'pow_tree_mappings', None)

        if tree_mappings is not None and cons.id in tree_mappings:
            tree = tree_mappings[cons.id]
            num_cones_per_col = dual_var.shape[1] // k
            w_duals, z_duals = _extract_pow_duals(
                tree, dual_var, n, k, num_cones_per_col
            )
        else:
            # n == 2: direct mapping, no tree decomposition
            w_duals = dual_var[:2, :]
            z_duals = dual_var[2, :]

        # Format as (n+1, k): W duals stacked above z duals
        return np.vstack([w_duals, z_duals[np.newaxis, :]])


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
        """Track auxiliary PSD constraint IDs for packed SOC dual recovery.

        Called by ExactCone2Cone.apply() after canonicalizing each SOC
        constraint. For packed SOC constraints (vector t), multiple PSD
        constraints are produced. This hook records the auxiliary PSD
        constraint IDs so that recover_dual() can reassemble the full
        SOC dual from all the PSD duals.
        """
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


class ComplexPSDConversion:
    """ComplexPSD -> PSD via 2n x 2n block matrix [[R, -I], [I, R]]."""
    source = ComplexPSD
    targets = {PSD}

    @staticmethod
    def canonicalize(con, args):
        real_part, imag_part = args
        block_matrix = cp.bmat([[real_part, -imag_part],
                                [imag_part, real_part]])
        return PSD(block_matrix), []

    @staticmethod
    def recover_dual(cons, dual_var, inverse_data, solution):
        n = cons.args[0].shape[0]
        # dual_var is (2n, 2n); extract complex n x n
        return dual_var[:n, :n] + 1j * dual_var[n:, :n]


class ExactCone2Cone(Canonicalization):

    CONVERSIONS = [PowNDConversion, SOCConversion, ComplexPSDConversion]

    CANON_METHODS = {c.source: c.canonicalize for c in CONVERSIONS}

    def __init__(self, problem=None, target_cones=None) -> None:
        conversions = self.CONVERSIONS
        if target_cones is not None:
            conversions = [c for c in conversions if c.source in target_cones]
        canon_methods = {c.source: c.canonicalize for c in conversions}
        self._dual_recovery = {c.source: c.recover_dual
                               for c in conversions if hasattr(c, 'recover_dual')}
        # _apply_hooks: per-conversion callbacks invoked by apply() right
        # after a constraint is canonicalized.  The flow is:
        #   1. apply() calls canonicalize_tree(constraint) to get
        #      (canon_constr, aux_constr).
        #   2. The hook for that constraint type is called with
        #      (original_constraint, canon_constr, aux_constr, inverse_data).
        #   3. The hook stores metadata on inverse_data (e.g.
        #      pow_tree_mappings for PowND, soc_packed_aux for packed SOC).
        #   4. Later, invert() calls recover_dual() which reads that
        #      metadata to reconstruct the original dual variables.
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

            # Call per-conversion hooks to record metadata (e.g. tree
            # structures or auxiliary constraint IDs) needed for dual
            # variable recovery in invert().
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
