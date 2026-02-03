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

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cone2cone.cone_tree import (
    LeafNode,
    SingleVarNode,
    SplitNode,
    TreeNode,
    get_root_cone_id,
)
from cvxpy.reductions.solution import Solution

EXOTIC_CONES = {
    PowConeND: {PowCone3D}
}
"""
^ An "exotic" cone is defined as any cone that isn't
supported by ParamConeProg. If ParamConeProg is updated
to support more cones, then it may be necessary to change
this file.
"""


def _build_pow_tree(
    W: Expression,
    alpha: np.ndarray,
    z: Expression,
    j: int,
    indices: list[int],
    alphas: list[float],
    x_3d: list[Expression],
    y_3d: list[Expression],
    z_3d: list[Expression],
    alpha_3d: list[float],
    is_root: bool = False
) -> tuple[Expression, float, TreeNode | None]:
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

        # Output variable
        if is_root:
            out_expr = z[j]
        else:
            out_expr = Variable(shape=())

        x_3d.append(W[i0, j])
        y_3d.append(W[i1, j])
        z_3d.append(out_expr)
        alpha_3d.append(r)

        # Build tree node only for first column
        tree = LeafNode(
            cone_id=len(x_3d) - 1,  # Index of this cone
            var_indices=(i0, i1)
        ) if j == 0 else None

        return out_expr, total, tree

    # Recursive case: split into two halves
    mid = n // 2
    left_indices = indices[:mid]
    right_indices = indices[mid:]
    left_alphas = alphas[:mid]
    right_alphas = alphas[mid:]

    # Recursively decompose each half
    left_expr, left_sum, left_tree = _build_pow_tree(
        W, alpha, z, j, left_indices, left_alphas,
        x_3d, y_3d, z_3d, alpha_3d, is_root=False
    )
    right_expr, right_sum, right_tree = _build_pow_tree(
        W, alpha, z, j, right_indices, right_alphas,
        x_3d, y_3d, z_3d, alpha_3d, is_root=False
    )

    # Combine with a new 3D cone
    total = left_sum + right_sum
    r = left_sum / total

    # Output variable
    if is_root:
        out_expr = z[j]
    else:
        out_expr = Variable(shape=())

    x_3d.append(left_expr)
    y_3d.append(right_expr)
    z_3d.append(out_expr)
    alpha_3d.append(r)

    # Build tree node only for first column
    if j == 0:
        tree = SplitNode(
            cone_id=len(x_3d) - 1,
            left=left_tree,
            right=right_tree
        )
    else:
        tree = None

    return out_expr, total, tree


def pow_nd_canon(con, args):
    """
    Canonicalize PowConeND to PowCone3D using a balanced binary tree decomposition.

    This reduces tree depth from O(n) to O(log n) compared to a linear chain,
    which can improve numerical stability and solver performance.

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
        # Direct mapping, no tree needed
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
        return can_con, [], None

    # Balanced tree decomposition for n > 2
    x_3d: list = []
    y_3d: list = []
    z_3d: list = []
    alpha_3d: list = []
    tree: TreeNode | None = None

    # Process each column
    for j in range(k):
        indices = list(range(n))
        alphas = list(alpha[:, j])

        _, _, col_tree = _build_pow_tree(
            W, alpha, z, j, indices, alphas,
            x_3d, y_3d, z_3d, alpha_3d, is_root=True
        )

        # Store tree from first column
        if j == 0:
            tree = col_tree

    # TODO: Ideally we should construct x,y,z,alpha_p3d by
    #   applying suitable sparse matrices to W,z,T, rather
    #   than using the hstack atom. (hstack will probably
    #   result in longer compile times).
    x_3d_expr = hstack(x_3d)
    y_3d_expr = hstack(y_3d)
    z_3d_expr = hstack(z_3d)
    alpha_p3d = hstack(alpha_3d)
    can_con = PowCone3D(x_3d_expr, y_3d_expr, z_3d_expr, alpha_p3d)

    # Return tree for dual variable recovery
    return can_con, [], tree


def _extract_pow_duals_recursive(
    node: TreeNode,
    parent_cone_id: int | None,
    is_left_child: bool,
    raw_duals: np.ndarray,
    w_duals: np.ndarray,
    j: int,
    num_cones_per_col: int,
) -> None:
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


def _extract_pow_duals(
    tree: TreeNode,
    raw_duals: np.ndarray,
    n: int,
    k: int,
    num_cones_per_col: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract W and z duals from raw 3D cone duals using tree structure."""
    w_duals = np.zeros((n, k))
    z_duals = np.zeros(k)

    root_idx = get_root_cone_id(tree)

    for j in range(k):
        _extract_pow_duals_recursive(
            tree, None, True,
            raw_duals, w_duals, j, num_cones_per_col
        )

        # Extract z dual from root cone
        if root_idx is not None:
            root_col = j * num_cones_per_col + root_idx
            z_duals[j] = raw_duals[2, root_col]

    return w_duals, z_duals


class Exotic2Common(Canonicalization):

    CANON_METHODS = {
        PowConeND: pow_nd_canon
    }

    def __init__(self, problem=None) -> None:
        super(Exotic2Common, self).__init__(
            problem=problem, canon_methods=Exotic2Common.CANON_METHODS)
        self._tree_mappings: dict[int, TreeNode] | None = None

    def canonicalize_expr(self, expr, args, canonicalize_params: bool = True):
        """Override to handle extra return value from pow_nd_canon."""
        if type(expr) in self.canon_methods:
            result = self.canon_methods[type(expr)](expr, args)
            # pow_nd_canon returns (can_con, [], tree)
            if isinstance(expr, PowConeND):
                can_expr, constraints, tree = result
                if tree is not None:
                    if self._tree_mappings is None:
                        self._tree_mappings = {}
                    self._tree_mappings[expr.id] = tree
                return can_expr, constraints
            return result
        return super().canonicalize_expr(expr, args, canonicalize_params)

    def apply(self, problem):
        """Override to copy tree mappings to inverse_data."""
        self._tree_mappings = None  # Reset for this canonicalization
        new_problem, inverse_data = super().apply(problem)
        # Copy tree mappings to inverse_data
        inverse_data.tree_mappings = self._tree_mappings
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

        tree_mappings: dict[int, TreeNode] | None = getattr(
            inverse_data, 'tree_mappings', None
        )

        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND):
                alpha, axis, _ = cons.get_data()
                n = alpha.shape[0] if axis == 0 else alpha.shape[1]
                k = cons.args[1].shape[0]  # Number of cones (columns)
                raw_duals = dvars[cons_id]  # Shape: (3, num_3d_cones)

                if tree_mappings is not None and cons_id in tree_mappings:
                    # Use tree structure for n > 2
                    tree = tree_mappings[cons_id]
                    num_cones_per_col = raw_duals.shape[1] // k
                    w_duals, z_duals = _extract_pow_duals(
                        tree, raw_duals, n, k, num_cones_per_col
                    )
                else:
                    # n == 2: direct mapping, no tree decomposition
                    # Raw duals shape is (3, k) where each column is one 3D cone
                    w_duals = raw_duals[:2, :]  # x and y duals are W[0] and W[1]
                    z_duals = raw_duals[2, :]   # z duals

                # Format as expected by save_dual_value: shape (n+1, k)
                # where [:-1, :] is W duals (n, k) and [-1, :] is z duals (k,)
                dvars[cons_id] = np.vstack([w_duals, z_duals[np.newaxis, :]])

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
