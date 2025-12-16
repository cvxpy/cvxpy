"""
Copyright 2025, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SOC dimension reduction to 3D cones.

This module provides the SOCDim3 reduction which converts arbitrary-dimensional
Second-Order Cone (SOC) constraints to equivalent systems of 3D SOC constraints.
This enables solvers that only support 3D SOC to handle arbitrary dimensional
SOC constraints.

The decomposition uses a binary tree structure:
- Dimension 1: Convert to NonNeg constraint (||[]|| <= t becomes t >= 0)
- Dimension 2: Convert to NonNeg constraints (|x| <= t becomes t-x >= 0, t+x >= 0)
- Dimension 3: Pass through unchanged
- Dimension 4: Chain of two 3D cones (special case for efficiency)
- Dimension n > 4: Binary split into balanced tree of 3D cones

Example:
    A dimension 5 cone ||(x1, x2, x3, x4)|| <= t becomes::

        ||(x1, x2)|| <= s1  (3D cone)
        ||(x3, x4)|| <= s2  (3D cone)
        ||(s1, s2)|| <= t   (3D cone)

    where s1, s2 are auxiliary variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from cvxpy.problems.problem import Problem

import numpy as np

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

# =============================================================================
# Tree Node Dataclasses
# =============================================================================

@dataclass(frozen=True)
class NonNegTree:
    """Tree node for cases that reduce to NonNeg constraints.

    - Dim-1 (||[]|| <= t): reduces to t >= 0
    - Dim-2 (|x| <= t): reduces to t-x >= 0, t+x >= 0

    Attributes
    ----------
    original_dim : int
        Original dimension (1 or 2).
    nonneg_ids : tuple
        Constraint IDs of the NonNeg constraints.
        For dim-1: (id of t >= 0,)
        For dim-2: (id of t-x >= 0, id of t+x >= 0)
    """
    original_dim: int = 1
    nonneg_ids: tuple = ()


@dataclass(frozen=True)
class LeafTree:
    """Tree node for a single 3D cone.

    Represents a leaf in the decomposition tree - an actual 3D SOC constraint.

    Attributes
    ----------
    cone_id : int
        The constraint ID of the 3D SOC cone.
    original_dim : int
        The original dimension of this cone (always 3 for LeafTree).
    """
    cone_id: int
    original_dim: int = 3


@dataclass(frozen=True)
class ChainTree:
    """Tree node for dim-4 special case: two chained cones.

    For dimension 4 (||(x1, x2, x3)|| <= t), we use a chain structure:
        ||(x1, x2)|| <= s
        ||(s, x3)|| <= t

    This avoids the unbalanced 1+2 split that would occur with generic binary split.

    Attributes
    ----------
    leaf_cone_id : int
        Constraint ID of the leaf cone (contains x1, x2).
    root_cone_id : int
        Constraint ID of the root cone (contains s, x3).
    original_dim : int
        Always 4 for chain decomposition.
    """
    leaf_cone_id: int
    root_cone_id: int
    original_dim: int = 4


@dataclass(frozen=True)
class SplitTree:
    """Tree node for binary split decomposition.

    For dimension n > 4, we split x into two halves and recursively decompose:
        ||x_left|| <= s1
        ||x_right|| <= s2
        ||(s1, s2)|| <= t

    Attributes
    ----------
    root_cone_id : int
        Constraint ID of the root cone (contains s1, s2).
    left : DecompositionTree
        Left subtree (decomposition of x_left).
    right : DecompositionTree
        Right subtree (decomposition of x_right).
    original_dim : int
        Original dimension of this cone.
    """
    root_cone_id: int
    left: DecompositionTree
    right: DecompositionTree
    original_dim: int


# Type alias for any tree node type
DecompositionTree = Union[NonNegTree, LeafTree, ChainTree, SplitTree]


# =============================================================================
# Inverse Data Dataclasses
# =============================================================================

@dataclass
class SOCTreeData:
    """Decomposition data for a single original SOC constraint.

    Attributes
    ----------
    trees : List[DecompositionTree]
        List of decomposition trees, one per cone in elementwise SOC.
    num_cones : int
        Number of individual cones (1 for simple SOC, >1 for elementwise).
    original_dim : int
        Dimension of each cone.
    axis : int
        Axis parameter from original SOC constraint.
    """
    trees: List[DecompositionTree]
    num_cones: int
    original_dim: int
    axis: int = 0


@dataclass
class SOCDim3InverseData:
    """Data needed to reconstruct original SOC duals from decomposed problem.

    Attributes
    ----------
    soc_trees : Dict[int, SOCTreeData]
        Mapping from original SOC constraint ID to its decomposition data.
    old_constraints : List
        Reference to original problem constraints.
    """
    soc_trees: Dict[int, SOCTreeData] = field(default_factory=dict)
    old_constraints: List = field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================

def _to_scalar_shape(expr: Expression) -> Expression:
    """Reshape expression to scalar form (1,) for SOC constructor.

    Parameters
    ----------
    expr : Expression
        Expression to reshape (should be scalar or shape (1,)).

    Returns
    -------
    Expression
        Expression with shape (1,).
    """
    return reshape(expr, (1,), order='F')


def _get_flat_dual(dual_value: Optional[Any]) -> Optional[np.ndarray]:
    """Convert a dual value to flat array format.

    Dual values can come in two formats:
    - Flat array: [lambda, mu1, mu2, ...] (from solver)
    - Split list: [lambda_array, mu_array] (after save_dual_value)

    This function normalizes to flat array format.

    Parameters
    ----------
    dual_value : Optional[Any]
        Dual value in either format, or None.

    Returns
    -------
    Optional[np.ndarray]
        Flat array [lambda, mu1, mu2, ...] or None if input is None.
    """
    if dual_value is None:
        return None

    # Check if it's a list with two elements (split format from save_dual_value)
    if isinstance(dual_value, list) and len(dual_value) == 2:
        t_val, x_val = dual_value
        if t_val is None or x_val is None:
            return None
        t_arr = np.atleast_1d(t_val)
        x_arr = np.atleast_1d(x_val).flatten(order='F')
        return np.concatenate([t_arr[:1], x_arr])

    # Otherwise assume it's already a flat array
    return np.atleast_1d(dual_value)


def _get_cone_ids_from_tree(tree: DecompositionTree) -> set:
    """Get all constraint IDs from a decomposition tree.

    Parameters
    ----------
    tree : DecompositionTree
        The decomposition tree to traverse.

    Returns
    -------
    set
        Set of all constraint IDs (SOC and NonNeg) in the tree.
    """
    if isinstance(tree, NonNegTree):
        return set(tree.nonneg_ids)
    elif isinstance(tree, LeafTree):
        return {tree.cone_id}
    elif isinstance(tree, ChainTree):
        return {tree.leaf_cone_id, tree.root_cone_id}
    elif isinstance(tree, SplitTree):
        ids = {tree.root_cone_id}
        ids.update(_get_cone_ids_from_tree(tree.left))
        ids.update(_get_cone_ids_from_tree(tree.right))
        return ids
    return set()


# =============================================================================
# Decomposition Functions
# =============================================================================

def _decompose_soc_single(
    t_expr: Expression,
    x_expr: Expression,
    soc3_out: List[SOC],
    nonneg_out: List[NonNeg]
) -> DecompositionTree:
    """Decompose a single ||x|| <= t constraint into tree of exactly 3D cones.

    This function recursively decomposes an n-dimensional SOC constraint into
    a tree of 3D SOC constraints using binary splitting.

    Parameters
    ----------
    t_expr : Expression
        The scalar bound expression (should have size 1).
    x_expr : Expression
        The vector argument (1D expression).
    soc3_out : List[SOC]
        Output list to append generated 3D SOC constraints to.
    nonneg_out : List[NonNeg]
        Output list to append generated NonNeg constraints to.

    Returns
    -------
    DecompositionTree
        Tree structure for dual variable reconstruction.

    Raises
    ------
    ValueError
        If t_expr or x_expr is None, or if t_expr is not scalar.
    """
    # Input validation
    if t_expr is None or x_expr is None:
        raise ValueError("t_expr and x_expr cannot be None")

    n = x_expr.size  # Number of elements in x

    # Dimension 1: ||[]|| <= t  ->  t >= 0 (degenerate case)
    if n == 0:
        c = NonNeg(t_expr)
        nonneg_out.append(c)
        return NonNegTree(original_dim=1, nonneg_ids=(c.id,))

    # Dimension 2: |x| <= t  ->  t - x >= 0, t + x >= 0 (no SOC needed)
    if n == 1:
        x_flat = x_expr.flatten(order='F')
        # |x| <= t is equivalent to: -t <= x <= t
        # Which is: t - x >= 0 AND t + x >= 0
        c1 = NonNeg(t_expr - x_flat)  # t - x >= 0
        c2 = NonNeg(t_expr + x_flat)  # t + x >= 0
        nonneg_out.extend([c1, c2])
        return NonNegTree(original_dim=2, nonneg_ids=(c1.id, c2.id))

    # Dimension 3: Already valid, pass through unchanged
    if n == 2:
        cone = SOC(_to_scalar_shape(t_expr), x_expr.flatten(order='F'))
        soc3_out.append(cone)
        return LeafTree(cone_id=cone.id)

    # Dimension 4: Special chain structure to avoid unbalanced 1+2 split
    # Decompose as: ||(x1, x2)|| <= s, ||(s, x3)|| <= t
    if n == 3:
        # Note: We use explicit NonNeg constraints rather than Variable(nonneg=True)
        # to make this reduction position-independent in the solving chain.
        # With nonneg=True, this reduction would need to come before CvxAttr2Constr.
        s = Variable()
        nonneg_out.append(NonNeg(s))
        x_left = x_expr[:2]  # First two elements
        x_last = x_expr[2]   # Last element

        # First cone: ||(x1, x2)|| <= s
        cone1 = SOC(_to_scalar_shape(s), x_left.flatten(order='F'))
        soc3_out.append(cone1)

        # Second cone: ||(s, x3)|| <= t
        root_x = vstack([s, x_last])
        cone2 = SOC(_to_scalar_shape(t_expr), root_x.flatten(order='F'))
        soc3_out.append(cone2)

        return ChainTree(
            leaf_cone_id=cone1.id,
            root_cone_id=cone2.id,
            original_dim=4
        )

    # n >= 4: Standard binary split
    # Split x into two halves and recursively decompose each
    mid = (n + 1) // 2  # Ceiling division, slightly favors left side
    x_left = x_expr[:mid]
    x_right = x_expr[mid:]

    # Create auxiliary variables for partial norms (with explicit NonNeg constraints)
    s1 = Variable()
    s2 = Variable()
    nonneg_out.append(NonNeg(s1))
    nonneg_out.append(NonNeg(s2))

    # Recursively decompose each half
    left_tree = _decompose_soc_single(s1, x_left.flatten(order='F'), soc3_out, nonneg_out)
    right_tree = _decompose_soc_single(s2, x_right.flatten(order='F'), soc3_out, nonneg_out)

    # Root constraint: ||(s1, s2)|| <= t (exactly 3D)
    root_x = vstack([s1, s2])
    root_cone = SOC(_to_scalar_shape(t_expr), root_x.flatten(order='F'))
    soc3_out.append(root_cone)

    return SplitTree(
        root_cone_id=root_cone.id,
        left=left_tree,
        right=right_tree,
        original_dim=n + 1
    )


# =============================================================================
# Dual Reconstruction Functions
# =============================================================================

def _collect_x_duals_into_array(
    tree: DecompositionTree,
    dual_vars: Dict[int, Any],
    out: np.ndarray,
    offset: int
) -> Optional[int]:
    """Collect x-component duals from leaf cones into pre-allocated array.

    For an SOC constraint ||x|| <= t, the dual is a flat array [lambda, mu1, mu2, ...]
    where lambda is the dual for t and mu is the dual for x.

    This function traverses the decomposition tree and writes all mu components
    from leaf cones into the output array in order.

    Parameters
    ----------
    tree : DecompositionTree
        The decomposition tree structure.
    dual_vars : Dict[int, Any]
        Mapping from constraint ID to dual variable value.
    out : np.ndarray
        Pre-allocated output array to write duals into.
    offset : int
        Current write position in the output array.

    Returns
    -------
    Optional[int]
        New offset after writing, or None if reconstruction failed.
    """
    if isinstance(tree, NonNegTree):
        if tree.original_dim == 1:
            # Dim-1: no x components
            return offset
        elif tree.original_dim == 2 and len(tree.nonneg_ids) == 2:
            # Dim-2: |x| <= t was converted to t-x >= 0, t+x >= 0
            # Duals: α for (t-x), β for (t+x)
            # x dual μ = β - α (coefficient of x in Lagrangian)
            alpha_raw = dual_vars.get(tree.nonneg_ids[0])
            beta_raw = dual_vars.get(tree.nonneg_ids[1])
            if alpha_raw is None or beta_raw is None:
                return None
            alpha = float(np.atleast_1d(alpha_raw).flat[0])
            beta = float(np.atleast_1d(beta_raw).flat[0])
            out[offset] = beta - alpha
            return offset + 1
        return offset

    if isinstance(tree, LeafTree):
        # Leaf cone: extract x-component duals
        dual_raw = dual_vars.get(tree.cone_id)
        if dual_raw is None:
            return None

        dual = _get_flat_dual(dual_raw)
        if dual is None or len(dual) < 2:
            return None

        # x components are all elements after the first (lambda)
        x_dual = dual[1:]

        # Write to output array
        n_write = len(x_dual)
        out[offset:offset + n_write] = x_dual
        return offset + n_write

    if isinstance(tree, ChainTree):
        # Chain structure for dim-4: ||(x1, x2)|| <= s, ||(s, x3)|| <= t
        leaf_dual_raw = dual_vars.get(tree.leaf_cone_id)
        root_dual_raw = dual_vars.get(tree.root_cone_id)

        if leaf_dual_raw is None or root_dual_raw is None:
            return None

        leaf_dual = _get_flat_dual(leaf_dual_raw)
        root_dual = _get_flat_dual(root_dual_raw)

        if leaf_dual is None or root_dual is None:
            return None

        # Get x1, x2 from leaf cone (elements 1, 2 of [lambda, mu1, mu2])
        if len(leaf_dual) >= 3:
            out[offset] = leaf_dual[1]
            out[offset + 1] = leaf_dual[2]
            offset += 2

        # Get x3 from root cone (element 2 of [lambda, s_dual, x3_dual])
        if len(root_dual) >= 3:
            out[offset] = root_dual[2]
            offset += 1

        return offset

    if isinstance(tree, SplitTree):
        # Binary split: collect from left and right subtrees
        new_offset = _collect_x_duals_into_array(tree.left, dual_vars, out, offset)
        if new_offset is None:
            return None
        return _collect_x_duals_into_array(tree.right, dual_vars, out, new_offset)

    return None


def _get_root_t_dual(tree: DecompositionTree, dual_vars: Dict[int, Any]) -> Optional[float]:
    """Get the t-component dual (lambda) from the root cone.

    Parameters
    ----------
    tree : DecompositionTree
        The decomposition tree structure.
    dual_vars : Dict[int, Any]
        Mapping from constraint ID to dual variable value.

    Returns
    -------
    Optional[float]
        The dual value for t from the root cone, or None if not available.
    """
    if isinstance(tree, NonNegTree):
        if tree.original_dim == 1:
            # Dim-1: just t >= 0, dual is the NonNeg dual
            if len(tree.nonneg_ids) >= 1:
                dual_raw = dual_vars.get(tree.nonneg_ids[0])
                if dual_raw is not None:
                    return float(np.atleast_1d(dual_raw).flat[0])
            return None
        elif tree.original_dim == 2 and len(tree.nonneg_ids) == 2:
            # Dim-2: |x| <= t was converted to t-x >= 0, t+x >= 0
            # Duals: α for (t-x), β for (t+x)
            # t dual λ = α + β (coefficient of t in Lagrangian)
            alpha_raw = dual_vars.get(tree.nonneg_ids[0])
            beta_raw = dual_vars.get(tree.nonneg_ids[1])
            if alpha_raw is None or beta_raw is None:
                return None
            alpha = float(np.atleast_1d(alpha_raw).flat[0])
            beta = float(np.atleast_1d(beta_raw).flat[0])
            return alpha + beta
        return None

    if isinstance(tree, LeafTree):
        # Single leaf is also root
        dual_raw = dual_vars.get(tree.cone_id)
        if dual_raw is None:
            return None
        dual = _get_flat_dual(dual_raw)
        return dual[0] if dual is not None and len(dual) >= 1 else None

    if isinstance(tree, ChainTree):
        # Root is the second cone
        dual_raw = dual_vars.get(tree.root_cone_id)
        if dual_raw is None:
            return None
        dual = _get_flat_dual(dual_raw)
        return dual[0] if dual is not None and len(dual) >= 1 else None

    if isinstance(tree, SplitTree):
        dual_raw = dual_vars.get(tree.root_cone_id)
        if dual_raw is None:
            return None
        dual = _get_flat_dual(dual_raw)
        return dual[0] if dual is not None and len(dual) >= 1 else None

    return None


def _reconstruct_soc_dual(
    tree: DecompositionTree,
    dual_vars: Dict[int, Any]
) -> Optional[np.ndarray]:
    """Reconstruct the original SOC dual from decomposed cone duals.

    For an SOC constraint ||x|| <= t, the dual is a flat array [lambda, mu1, mu2, ...]
    where lambda is the dual for t and mu is the dual for x.

    This function reconstructs the original dual by:
    1. Getting lambda from the root cone
    2. Collecting all mu components from leaf cones in order

    Parameters
    ----------
    tree : DecompositionTree
        The decomposition tree structure.
    dual_vars : Dict[int, Any]
        Mapping from constraint ID to dual variable value (flat arrays).

    Returns
    -------
    Optional[np.ndarray]
        Reconstructed dual as flat array [lambda, mu1, mu2, ...] or None if failed.
    """
    t_dual = _get_root_t_dual(tree, dual_vars)
    if t_dual is None:
        return None

    # Calculate x dimension from tree's original_dim
    x_size = tree.original_dim - 1
    if x_size == 0:
        return np.array([t_dual])

    # Pre-allocate array for x duals
    x_duals = np.empty(x_size)
    final_offset = _collect_x_duals_into_array(tree, dual_vars, x_duals, 0)

    if final_offset is None:
        return None

    return np.concatenate([[t_dual], x_duals])


# =============================================================================
# Main Reduction Class
# =============================================================================

class SOCDim3(Reduction):
    """Convert n-dimensional SOC constraints to dimension-3 SOC constraints.

    This reduction enables solvers that only support 3D SOC to handle
    arbitrary dimensional SOC constraints.

    The decomposition uses a binary tree structure:

    - Dimension 1 (||[]|| <= t): Convert to NonNeg(t)
    - Dimension 2 (|x| <= t): Pad with auxiliary variable to 3D
    - Dimension 3: Pass through unchanged
    - Dimension 4: Special chain structure (two 3D cones)
    - Dimension n > 4: Recursively split into tree of 3D cones

    Example
    -------
    >>> x = cp.Variable(4)
    >>> t = cp.Variable(nonneg=True)
    >>> prob = cp.Problem(cp.Minimize(t), [cp.norm(x, 2) <= t])
    >>> reduction = SOCDim3()
    >>> new_prob, inv_data = reduction.apply(prob)
    >>> # new_prob has only 3D SOC constraints
    """

    def accepts(self, problem: Problem) -> bool:
        """Check if this reduction accepts the given problem.

        This reduction accepts any problem.

        Parameters
        ----------
        problem : Problem
            The optimization problem.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def apply(self, problem: Problem) -> Tuple[Problem, SOCDim3InverseData]:
        """Apply SOCDim3 decomposition to all SOC constraints.

        Parameters
        ----------
        problem : Problem
            The optimization problem to transform.

        Returns
        -------
        new_problem : Problem
            Problem with all SOC constraints decomposed to dim-3.
        inverse_data : SOCDim3InverseData
            Data needed to reconstruct original dual variables.
        """
        inverse_data = SOCDim3InverseData(
            soc_trees={},
            old_constraints=problem.constraints
        )

        # Check if there are any SOC constraints to decompose
        has_soc = any(isinstance(c, SOC) for c in problem.constraints)
        if not has_soc:
            return problem, inverse_data

        new_constraints: List = []

        for con in problem.constraints:
            if isinstance(con, SOC):
                # Decompose this SOC constraint
                t = con.args[0]
                X = con.args[1]
                axis = con.axis

                # Handle axis: if axis == 1, transpose X to work with columns
                if axis == 1:
                    X = X.T

                # Get dimensions
                if len(X.shape) == 1:
                    # Single cone case
                    X_reshaped = reshape(X, (X.size, 1), order='F')
                    t_reshaped = _to_scalar_shape(t)
                else:
                    X_reshaped = X
                    t_reshaped = t

                num_cones = t_reshaped.size

                # Validate dimensions match
                if X_reshaped.shape[1] != num_cones and num_cones > 1:
                    raise ValueError(
                        f"Dimension mismatch: t has {num_cones} elements but "
                        f"X has {X_reshaped.shape[1]} columns"
                    )

                all_trees: List[DecompositionTree] = []
                soc3_out: List[SOC] = []
                nonneg_out: List[NonNeg] = []

                # Note on vectorization: This loop processes each cone sequentially.
                # TODO add vectorization.
                for i in range(num_cones):
                    t_i = t_reshaped[i] if num_cones > 1 else t_reshaped[0]
                    x_i = X_reshaped[:, i] if num_cones > 1 else X_reshaped[:, 0]

                    tree = _decompose_soc_single(t_i, x_i, soc3_out, nonneg_out)
                    all_trees.append(tree)

                new_constraints.extend(soc3_out)
                new_constraints.extend(nonneg_out)

                # Store tree structure for dual reconstruction
                cone_sizes = con.cone_sizes()
                inverse_data.soc_trees[con.id] = SOCTreeData(
                    trees=all_trees,
                    num_cones=num_cones,
                    original_dim=cone_sizes[0] if cone_sizes else 3,
                    axis=axis
                )
            else:
                # Non-SOC constraint: pass through
                new_constraints.append(con)

        new_problem = cvxtypes.problem()(problem.objective, new_constraints)
        return new_problem, inverse_data

    def invert(
        self,
        solution: Solution,
        inverse_data: SOCDim3InverseData
    ) -> Solution:
        """Reconstruct solution with original SOC dual variables.

        Parameters
        ----------
        solution : Solution
            Solution from the decomposed problem.
        inverse_data : SOCDim3InverseData
            Data from apply() containing tree structures.

        Returns
        -------
        Solution
            Solution with reconstructed dual variables.
        """
        # Pass through primal variables unchanged
        pvars = solution.primal_vars.copy() if solution.primal_vars else {}

        # Handle failed solutions early
        if not solution.dual_vars:
            return Solution(solution.status, solution.opt_val, pvars, {}, solution.attr)

        # Reconstruct dual variables
        dvars: Dict[int, Any] = {}

        # Identify which constraint IDs belong to decomposed cones
        decomposed_cone_ids: set = set()
        for tree_data in inverse_data.soc_trees.values():
            for tree in tree_data.trees:
                decomposed_cone_ids.update(_get_cone_ids_from_tree(tree))

        # Copy non-decomposed duals
        for cid, dual in solution.dual_vars.items():
            if cid not in decomposed_cone_ids:
                dvars[cid] = dual

        # Reconstruct SOC duals
        for orig_id, tree_data in inverse_data.soc_trees.items():
            trees = tree_data.trees
            if len(trees) == 1:
                # Single cone - returns flat array [lambda, mu1, mu2, ...]
                reconstructed = _reconstruct_soc_dual(trees[0], solution.dual_vars)
                if reconstructed is not None:
                    dvars[orig_id] = reconstructed
            else:
                # Multiple cones (elementwise SOC)
                # Reconstruct each cone's dual as flat array [t, x1, x2, ...]
                all_duals: List[np.ndarray] = []
                success = True
                for tree in trees:
                    reconstructed = _reconstruct_soc_dual(tree, solution.dual_vars)
                    if reconstructed is None:
                        success = False
                        break
                    all_duals.append(reconstructed)

                if success:
                    # CVXPY SOC expects duals as [t_array, x_array] for elementwise
                    # Extract t (first element) and x (rest) from each cone
                    t_duals = np.array([d[0] for d in all_duals])
                    x_duals = np.column_stack([d[1:] for d in all_duals])
                    # For axis=1, x_duals needs to be transposed
                    if tree_data.axis == 1:
                        x_duals = x_duals.T
                    dvars[orig_id] = [t_duals, x_duals]

        return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)
