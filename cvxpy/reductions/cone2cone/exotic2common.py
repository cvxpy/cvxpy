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
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
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
    tree_mapping = None  # Will be set for n > 2
    if n == 2:
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        # Balanced tree decomposition
        # We need n-2 auxiliary variables total (same as linear chain)
        x_3d, y_3d, z_3d, alpha_3d = [], [], [], []
        aux_vars = []  # Will hold all auxiliary variables created

        # Track mapping from W indices to 3D cone positions for dual recovery
        # w_to_cone[i] = (cone_index, 'x' or 'y') means W[i] appears in that position
        # We track per-column, but all columns have the same structure
        cone_counter = 0
        w_to_cone = {}

        def decompose(indices, alphas, j):
            """
            Recursively decompose indices into a balanced binary tree.

            indices: list of row indices into W for column j
            alphas: corresponding alpha values (same length as indices)
            j: column index

            Returns: (expr, alpha_sum) where expr is either W[i,j] or an aux variable
                     representing prod(W[indices]^(alphas/alpha_sum))
            """
            nonlocal cone_counter

            if len(indices) == 1:
                # Base case: single variable, no cone needed
                return W[indices[0], j], alphas[0]

            if len(indices) == 2:
                # Base case: two variables, create one 3D cone
                i0, i1 = indices
                a0, a1 = alphas
                total = a0 + a1
                r = a0 / total

                # Create auxiliary variable for output
                aux = Variable(shape=())
                aux_vars.append(aux)

                # Track mapping (only on first column since structure is same)
                if j == 0:
                    w_to_cone[i0] = (cone_counter, 'x')
                    w_to_cone[i1] = (cone_counter, 'y')
                    cone_counter += 1

                x_3d.append(W[i0, j])
                y_3d.append(W[i1, j])
                z_3d.append(aux)
                alpha_3d.append(r)

                return aux, total

            # Recursive case: split into two halves
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]
            left_alphas = alphas[:mid]
            right_alphas = alphas[mid:]

            # Recursively decompose each half
            left_expr, left_sum = decompose(left_indices, left_alphas, j)
            right_expr, right_sum = decompose(right_indices, right_alphas, j)

            # Combine with a new 3D cone
            total = left_sum + right_sum
            r = left_sum / total

            # Create auxiliary variable for output
            aux = Variable(shape=())
            aux_vars.append(aux)

            # Track cone counter for intermediate cones (only on first column)
            if j == 0:
                cone_counter += 1

            x_3d.append(left_expr)
            y_3d.append(right_expr)
            z_3d.append(aux)
            alpha_3d.append(r)

            return aux, total

        # Process each column
        for j in range(k):
            indices = list(range(n))
            alphas = list(alpha[:, j])

            # Build balanced tree, but the root outputs to z[j] instead of aux var
            if n == 2:
                # Already handled above
                pass
            else:
                # Split at root level
                mid = n // 2
                left_indices = indices[:mid]
                right_indices = indices[mid:]
                left_alphas = alphas[:mid]
                right_alphas = alphas[mid:]

                # Recursively decompose each half
                left_expr, left_sum = decompose(left_indices, left_alphas, j)
                right_expr, right_sum = decompose(right_indices, right_alphas, j)

                # Root cone outputs to z[j]
                total = left_sum + right_sum
                r = left_sum / total

                x_3d.append(left_expr)
                y_3d.append(right_expr)
                z_3d.append(z[j])
                alpha_3d.append(r)

        # TODO: Ideally we should construct x,y,z,alpha_p3d by
        #   applying suitable sparse matrices to W,z,T, rather
        #   than using the hstack atom. (hstack will probably
        #   result in longer compile times).
        x_3d = hstack(x_3d)
        y_3d = hstack(y_3d)
        z_3d = hstack(z_3d)
        alpha_p3d = hstack(alpha_3d)
        can_con = PowCone3D(x_3d, y_3d, z_3d, alpha_p3d)

        # Return tree mapping for dual variable recovery:
        # w_to_cone: dict mapping W index to (cone_index, 'x' or 'y')
        # root_cone_index: index of root cone (whose z is the original z)
        # num_cones: total cones per column
        tree_mapping = (w_to_cone, cone_counter, cone_counter + 1)
    # Return a single PowCone3D constraint defined over all auxiliary
    # variables needed for the reduction to go through.
    # There are no "auxiliary constraints" beyond this one.
    # Third element is tree_mapping for dual recovery (None for n==2).
    return can_con, [], tree_mapping if n > 2 else None


class Exotic2Common(Canonicalization):

    CANON_METHODS = {
        PowConeND: pow_nd_canon
    }

    def __init__(self, problem=None) -> None:
        super(Exotic2Common, self).__init__(
            problem=problem, canon_methods=Exotic2Common.CANON_METHODS)
        self._tree_mappings = None  # Temporarily store mappings during canonicalization

    def canonicalize_expr(self, expr, args, canonicalize_params: bool = True):
        """Override to handle extra return value from pow_nd_canon."""
        if type(expr) in self.canon_methods:
            result = self.canon_methods[type(expr)](expr, args)
            # pow_nd_canon returns (can_con, [], tree_mapping)
            if len(result) == 3:
                can_expr, constraints, tree_mapping = result
                if tree_mapping is not None:
                    if self._tree_mappings is None:
                        self._tree_mappings = {}
                    self._tree_mappings[expr.id] = tree_mapping
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
            #NOTE: pre-maturely trigger return of the method in case the problem
            # is infeasible (otherwise will run into some opaque errors)
            return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

        tree_mappings = getattr(inverse_data, 'tree_mappings', None)

        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND):
                alpha, axis, _ = cons.get_data()
                n = alpha.shape[0] if axis == 0 else alpha.shape[1]
                k = cons.args[1].shape[0]  # Number of cones (columns)
                raw_duals = dvars[cons_id]  # Shape: (3, num_3d_cones)

                if tree_mappings is not None and cons_id in tree_mappings:
                    # Use balanced tree mapping for n > 2
                    w_to_cone, root_idx, num_cones = tree_mappings[cons_id]
                    w_duals = np.zeros((n, k))
                    z_duals = np.zeros(k)

                    for j in range(k):
                        # Extract W duals from leaf cones
                        for w_idx, (cone_idx, pos) in w_to_cone.items():
                            col = j * num_cones + cone_idx
                            row = 0 if pos == 'x' else 1
                            w_duals[w_idx, j] = raw_duals[row, col]

                        # Extract z dual from root cone
                        root_col = j * num_cones + root_idx
                        z_duals[j] = raw_duals[2, root_col]
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
