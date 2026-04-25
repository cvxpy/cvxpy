"""
Copyright, the CVXPY authors

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

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cvxpy.constraints import (
    PSD,
    SOC,
    ExpCone,
    NonNeg,
    PowCone3D,
    PowConeND,
    SvecPSD,
    Zero,
)
from cvxpy.lin_ops.lin_op import CONSTANT_ID
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.reduction import Reduction


class ExtractIdentityCones(Reduction):
    """Move slack-side cone constraints whose A block is identity onto
    a subvector of the primal variable via ``problem.x_cones``.

    For each candidate constraint ``c`` whose materialised A block has
    rows equal to ``e_j^T`` with distinct ``j`` and whose constant
    offset is zero, the rows are dropped from the parameter tensor and
    a tuple ``('nonneg', x_indices)`` (or ``'soc'``) is appended to
    ``problem.x_cones``.  The slack-side cone is removed from
    ``cone_dims``.

    Position in chain::

        ... -> ConeMatrixStuffing -> ExtractIdentityCones -> Solver

    DPP correctness
    ---------------
    The structural check inspects the parameter tensor directly: for
    each candidate output row, every entry of the tensor row must be
    zero **except** the column for ``CONSTANT_ID``.  That guarantees
    the row is constant under any parameter assignment, so the cached
    canonicalisation is safe across reparameterisations.

    Solver opt-in
    -------------
    Only the cone kinds listed in ``solver_context.x_cone_kinds`` are
    extracted.  A solver without an x_cone path advertises an empty
    set and this reduction is a no-op.
    """

    def __init__(self, solver_context=None) -> None:
        kinds = (set(solver_context.x_cone_kinds)
                 if solver_context is not None else set())
        # Only kinds we know how to detect identity-pattern for.
        self._kinds = kinds & {'nonneg', 'soc'}

    def accepts(self, problem) -> bool:
        if not isinstance(problem, ParamConeProg):
            return False
        if problem.is_mixed_integer():
            return False
        if problem.x_cones:
            return False
        if not self._kinds:
            return False
        return True

    def apply(self, problem):
        if not self.accepts(problem):
            return problem, _NoopInverse()

        m_total = problem.constr_size
        n = problem.x.size
        const_col = problem.param_id_to_col.get(CONSTANT_ID, None)
        if const_col is None:
            return problem, _NoopInverse()

        # Drive everything off ReducedMat: reduced_mat is shape
        # (nnz_M, p+1) and problem_data_index = (indices, indptr, shape)
        # is the CSC pattern of the materialised M (column-major over
        # the n+1 output columns).  We never materialise or slice the
        # raw (m_total*(n+1)) × (p+1) tensor.
        problem.reduced_A.cache()
        reduced_mat = problem.reduced_A.reduced_mat
        problem_data_index = problem.reduced_A.problem_data_index
        if reduced_mat is None or problem_data_index is None:
            return problem, _NoopInverse()
        rm_csr = reduced_mat.tocsr() if not sp.issparse(reduced_mat) or \
            reduced_mat.format != 'csr' else reduced_mat
        indices, indptr, _ = problem_data_index
        # M-row and M-col for each reduced_mat row k.
        row_of_k = np.asarray(indices)
        col_of_k = np.repeat(
            np.arange(n + 1, dtype=np.int64), np.diff(indptr)
        )

        constraints = list(problem.constraints)
        constr_map = problem.constr_map

        # Walk constraints once, recording each candidate's row range
        # plus a per-row "candidate id" lookup (-1 if non-candidate).
        # Each call to _try_extract_block then operates only on the ks
        # already grouped to that candidate, instead of scanning the
        # whole reduced_mat.
        block_id_of_row = np.full(m_total, -1, dtype=np.int64)
        candidates = []  # ordered list of (kind, c, r0, r1)
        ordered_constraints = []  # (kind, c, r0, r1) for every constraint
        row = 0
        for c in constr_map.get(Zero, []):
            ordered_constraints.append(('zero_pass', c, row, row + c.size))
            row += c.size
        for c in constr_map.get(NonNeg, []):
            r0, r1 = row, row + c.size
            if 'nonneg' in self._kinds:
                block_id_of_row[r0:r1] = len(candidates)
                candidates.append(('nonneg', c, r0, r1))
                ordered_constraints.append(('nonneg_cand', c, r0, r1))
            else:
                ordered_constraints.append(('pass', c, r0, r1))
            row = r1
        for c in constr_map.get(SOC, []):
            r0, r1 = row, row + c.size
            if 'soc' in self._kinds:
                block_id_of_row[r0:r1] = len(candidates)
                candidates.append(('soc', c, r0, r1))
                ordered_constraints.append(('soc_cand', c, r0, r1))
            else:
                ordered_constraints.append(('pass', c, r0, r1))
            row = r1
        for cone_type in (PSD, SvecPSD, ExpCone, PowCone3D, PowConeND):
            for c in constr_map.get(cone_type, []):
                ordered_constraints.append(('pass', c, row, row + c.size))
                row += c.size

        if not candidates:
            return problem, _NoopInverse()

        # Group candidate ks by block id once, sorted by row within
        # each group.  Replaces 300+ O(nnz) range-mask ops with one
        # O(nnz log nnz) sort.
        block_id_of_k = block_id_of_row[row_of_k]
        candidate_mask = block_id_of_k >= 0
        candidate_k = np.where(candidate_mask)[0]
        if candidate_k.size == 0:
            return problem, _NoopInverse()
        sort_order = np.lexsort((row_of_k[candidate_k],
                                 block_id_of_k[candidate_k]))
        sorted_ks = candidate_k[sort_order]
        sorted_block_ids = block_id_of_k[sorted_ks]
        sorted_rows = row_of_k[sorted_ks]
        sorted_cols = col_of_k[sorted_ks]
        boundaries = np.searchsorted(
            sorted_block_ids, np.arange(len(candidates) + 1)
        )

        rm_indptr = rm_csr.indptr
        rm_indices = rm_csr.indices
        rm_data = rm_csr.data

        # Global per-k checks (vectorised across all candidate ks):
        #   - reduced_mat row has exactly one nonzero, in const_col,
        #     with value +1 (structural constancy + identity value)
        #   - the M column is in the A part (j < n)
        nnz_per_row = rm_indptr[sorted_ks + 1] - rm_indptr[sorted_ks]
        first_col = rm_indices[rm_indptr[sorted_ks]]
        first_val = rm_data[rm_indptr[sorted_ks]]
        bad_k = (
            (nnz_per_row != 1)
            | (first_col != const_col)
            | (first_val != 1.0)
            | (sorted_cols >= n)
        )
        # Per-block "any-bad" via reduceat over the block boundaries.
        bad_block = np.zeros(len(candidates), dtype=bool)
        nz_seg = boundaries[1:] - boundaries[:-1]
        nonempty = nz_seg > 0
        if nonempty.any():
            seg_starts = boundaries[:-1][nonempty]
            seg_bad = np.add.reduceat(bad_k.astype(np.int64), seg_starts) > 0
            bad_block[nonempty] = seg_bad

        # Decide each candidate's fate (extract or pass through).
        # Global checks already reject malformed entries; per-block
        # work here is only the row-permutation check (rows must be
        # exactly [r0, r1)) and a distinct-x-slot reservation.
        x_used_mask = np.zeros(n, dtype=bool)
        extract_results: list = [None] * len(candidates)
        for b in range(len(candidates)):
            if bad_block[b]:
                continue
            r0, r1 = candidates[b][2], candidates[b][3]
            start, end = int(boundaries[b]), int(boundaries[b + 1])
            block_size = r1 - r0
            if end - start != block_size:
                continue
            block_rows = sorted_rows[start:end]
            # rows are sorted (by lexsort) — endpoints + length are
            # enough to confirm a permutation of [r0, r1).
            if block_rows[0] != r0 or block_rows[-1] != r1 - 1:
                continue
            block_cols = sorted_cols[start:end]
            # Distinct columns within the block, and disjoint from
            # already-claimed x slots.
            if x_used_mask[block_cols].any():
                continue
            if np.unique(block_cols).size != block_size:
                continue
            extract_results[b] = block_cols.tolist()
            x_used_mask[block_cols] = True

        # Materialise extracted entries and the new constraint list.
        extracted = []
        new_constraints = []
        for tag, c, r0, r1 in ordered_constraints:
            if tag == 'zero_pass' or tag == 'pass':
                new_constraints.append(c)
                continue
            # tag in ('nonneg_cand', 'soc_cand') — find its candidate id.
            b = int(block_id_of_row[r0])
            x_idx = extract_results[b]
            if x_idx is None:
                new_constraints.append(c)
            elif tag == 'nonneg_cand':
                extracted.append(('nonneg', x_idx, c.id, (r0, r1), c.size))
            else:  # soc_cand
                extracted.append(('soc', x_idx, c.id, (r0, r1), c.size,
                                  tuple(c.cone_sizes())))

        if not extracted:
            return problem, _NoopInverse()

        # Build kept-row mask and the kept_pos lookup (i_old -> i_new).
        kept_M_rows = np.ones(m_total, dtype=bool)
        for _, _, _, (r0, r1), *_ in extracted:
            kept_M_rows[r0:r1] = False
        kept_M_idx = np.where(kept_M_rows)[0]
        m_kept = len(kept_M_idx)
        kept_pos = np.full(m_total, -1, dtype=np.int64)
        kept_pos[kept_M_idx] = np.arange(m_kept)

        # Rebuild the raw param tensor (matrix_data for the new
        # ParamConeProg) from reduced_mat — only the kept reduced
        # rows survive, with their tensor positions remapped to the
        # new (smaller) M.  This is O(nnz_M), no huge slicing.
        keep_k = kept_pos[row_of_k] >= 0
        kept_ks = np.where(keep_k)[0]
        new_i_M = kept_pos[row_of_k[kept_ks]]
        new_j_M = col_of_k[kept_ks]
        new_tensor_rows = new_i_M + new_j_M * m_kept

        kept_reduced = rm_csr[kept_ks, :].tocoo()
        new_A_rows = new_tensor_rows[kept_reduced.row]
        new_A_cols = kept_reduced.col
        new_A_data = kept_reduced.data
        new_A_shape = (m_kept * (n + 1), problem.A.shape[1])
        new_A = sp.csc_array(
            sp.coo_array(
                (new_A_data, (new_A_rows, new_A_cols)), shape=new_A_shape,
            )
        )

        x_cones = [(kind, list(idx), constr_id)
                   for kind, idx, constr_id, *_ in extracted]

        new_pcp = ParamConeProg(
            problem.q,
            problem.x,
            new_A,
            problem.variables,
            problem.var_id_to_col,
            new_constraints,
            problem.parameters,
            problem.param_id_to_col,
            P=problem.P,
            formatted=True,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            lb_tensor=problem.lb_tensor,
            ub_tensor=problem.ub_tensor,
            x_cones=x_cones,
        )

        inv = {
            'extracted': extracted,
            'kept_M_idx': kept_M_idx,
            'orig_constr_ids': [c.id for c in constraints],
        }
        return new_pcp, inv

    def invert(self, solution, inverse_data):
        """Restore extracted-block duals into the Solution.

        The solver is responsible for computing x_cone dual variables
        (it has the materialised q, A, z) and for placing them in
        ``solution.dual_vars`` keyed by the *original* constraint id.
        That keeps the reduction's invert simple — it only forwards.
        """
        if isinstance(inverse_data, _NoopInverse):
            return solution
        # The solver stored extracted-cone duals under the original
        # constraint ids already, so we just pass through.
        return solution


class _NoopInverse:
    """Sentinel inverse-data when accepts() returned False or no
    block was extractable; ``invert`` becomes a pass-through."""
    pass
