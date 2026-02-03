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

Tree node types for balanced binary tree decomposition of cones.
Used by SOCDim3 and Exotic2Common reductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeafNode:
    """Leaf node: a single 3D cone with original variable indices."""
    cone_id: int
    var_indices: tuple[int, ...]


@dataclass(frozen=True)
class SplitNode:
    """Internal node combining two subtrees with a 3D cone."""
    cone_id: int
    left: TreeNode
    right: TreeNode


@dataclass(frozen=True)
class SingleVarNode:
    """A single variable with no associated cone (n=1 base case in tree decomposition)."""
    var_index: int


@dataclass(frozen=True)
class SpecialNode:
    """Node for special cases (e.g., SOC dim-1, dim-2, dim-4)."""
    node_type: str
    cone_ids: tuple[int, ...]
    var_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


TreeNode = LeafNode | SplitNode | SingleVarNode | SpecialNode


def get_all_cone_ids(node: TreeNode) -> set[int]:
    """Get all constraint IDs from a tree."""
    if isinstance(node, LeafNode):
        return {node.cone_id}
    elif isinstance(node, SplitNode):
        ids = {node.cone_id}
        ids.update(get_all_cone_ids(node.left))
        ids.update(get_all_cone_ids(node.right))
        return ids
    elif isinstance(node, SingleVarNode):
        return set()
    elif isinstance(node, SpecialNode):
        return set(node.cone_ids)
    return set()


def get_leaf_nodes(node: TreeNode) -> list[LeafNode]:
    """Get all leaf nodes in left-to-right order."""
    if isinstance(node, LeafNode):
        return [node]
    elif isinstance(node, SplitNode):
        return get_leaf_nodes(node.left) + get_leaf_nodes(node.right)
    elif isinstance(node, SingleVarNode):
        return []
    elif isinstance(node, SpecialNode):
        return []
    return []


def get_root_cone_id(node: TreeNode) -> int | None:
    """Get the constraint ID of the root cone."""
    if isinstance(node, LeafNode):
        return node.cone_id
    elif isinstance(node, SplitNode):
        return node.cone_id
    elif isinstance(node, SpecialNode):
        return node.cone_ids[-1] if node.cone_ids else None
    return None
