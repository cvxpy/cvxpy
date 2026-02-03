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
    if n == 2:
        can_con = PowCone3D(W[0, :], W[1, :], z, alpha[0, :])
    else:
        # Balanced tree decomposition
        # We need n-2 auxiliary variables total (same as linear chain)
        x_3d, y_3d, z_3d, alpha_3d = [], [], [], []
        aux_vars = []  # Will hold all auxiliary variables created

        def decompose(indices, alphas, j):
            """
            Recursively decompose indices into a balanced binary tree.

            indices: list of row indices into W for column j
            alphas: corresponding alpha values (same length as indices)
            j: column index

            Returns: (expr, alpha_sum) where expr is either W[i,j] or an aux variable
                     representing prod(W[indices]^(alphas/alpha_sum))
            """
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
    # Return a single PowCone3D constraint defined over all auxiliary
    # variables needed for the reduction to go through.
    # There are no "auxiliary constraints" beyond this one.
    return can_con, []


class Exotic2Common(Canonicalization):

    CANON_METHODS = {
        PowConeND: pow_nd_canon
    }

    def __init__(self, problem=None) -> None:
        super(Exotic2Common, self).__init__(
            problem=problem, canon_methods=Exotic2Common.CANON_METHODS)

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

        dv = {}
        for cons_id, cons in inverse_data.id2cons.items():
            if isinstance(cons, PowConeND):
                div_size = int(dvars[cons_id].shape[1] // cons.args[1].shape[0])
                dv[cons_id] = []
                for i in range(cons.args[1].shape[0]):
                    # Iterating over the vectorized constraints
                    dv[cons_id].append([])
                    tmp_duals = dvars[cons_id][:, i * div_size: (i + 1) * div_size]
                    for j, col_dvars in enumerate(tmp_duals.T):
                        if j == len(tmp_duals.T) - 1:
                            dv[cons_id][-1] += [col_dvars[0], col_dvars[1]]
                        else:
                            dv[cons_id][-1].append(col_dvars[0])
                    dv[cons_id][-1].append(tmp_duals.T[0][-1]) # dual value corresponding to `z`
                dvars[cons_id] = np.array(dv[cons_id])

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)
