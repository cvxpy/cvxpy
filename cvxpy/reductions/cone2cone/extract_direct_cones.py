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

from cvxpy.constraints import SOC, ExpCone, NonNeg, PowCone3D, PowConeND, SvecPSD
from cvxpy.lin_ops.lin_op import CONSTANT_ID
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import DirectCone, ParamConeProg
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


def _unit_cones(constr, indices, values, kind, sizes, extras) -> list[DirectCone]:
    if np.any(values != 1):
        return []
    cones = []
    start = 0
    for size, extra in zip(sizes, extras):
        cones.append(DirectCone(kind, indices[start:start + size].tolist(), constr.id, extra))
        start += size
    return cones


def _extract_nonneg(constr, indices, values) -> list[DirectCone]:
    return _unit_cones(constr, indices, values, 'nonneg', [constr.size], [{}])


def _extract_soc(constr, indices, values) -> list[DirectCone]:
    return _unit_cones(constr, indices, values, 'soc', constr.cone_sizes(),
                       [{}] * constr.num_cones())


def _extract_psd(constr, indices, values) -> list[DirectCone]:
    # Direct PSD indices refer to unscaled upper-triangle entries, while
    # SvecPSD rows multiply off-diagonal entries by sqrt(2).
    n = constr._n
    size = n * (n + 1) // 2
    pattern = np.full(size, np.sqrt(2))
    j = np.arange(n)
    pattern[j * (j + 3) // 2] = 1
    if np.any(values != np.tile(pattern, constr.num_cones())):
        return []
    return [DirectCone('psd_triangle', idx.tolist(), constr.id, {'psd_k': n})
            for idx in indices.reshape(-1, size)]


def _extract_exp(constr, indices, values) -> list[DirectCone]:
    count = constr.num_cones()
    return _unit_cones(constr, indices, values, 'exp', [3] * count, [{}] * count)


def _extract_power(constr, indices, values) -> list[DirectCone]:
    # As in ConeDims, alpha is snapshotted; parameterized alpha is not DPP.
    extras = [{'alpha': float(a)} for a in np.asarray(constr.alpha.value).ravel()]
    return _unit_cones(constr, indices, values, 'power', [3] * constr.num_cones(), extras)


def _extract_gen_power(constr, indices, values) -> list[DirectCone]:
    # ConeMatrixStuffing normalizes PowConeND to axis=0.
    alpha = np.asarray(constr.alpha.value).reshape(-1, constr.num_cones())
    extras = [{'alphas': a.tolist(), 'dim2': 1} for a in alpha.T]
    return _unit_cones(constr, indices, values, 'gen_power', constr.cone_sizes(), extras)


def _identity_rows(problem, reduced, rows, cols):
    """Find structurally constant single-entry rows in one linear pass.

    Each reduced tensor row represents one entry of [A b]. Counting these
    entries by output row avoids sorting or rescanning the tensor per cone.
    Parameter-dependent entries remain ineligible even if their current
    value happens to be zero or one.
    """
    m, n = problem.constr_size, problem.x.size
    x_indices = np.full(m, -1, dtype=np.int64)
    values = np.zeros(m)
    counts = np.bincount(rows, minlength=m)
    single = np.flatnonzero((counts[rows] == 1) & (np.diff(reduced.indptr) == 1))
    positions = reduced.indptr[single]
    valid = ((reduced.indices[positions] == problem.param_id_to_col[CONSTANT_ID])
             & (cols[single] < n))
    single, positions = single[valid], positions[valid]
    x_indices[rows[single]] = cols[single]
    values[rows[single]] = reduced.data[positions]
    return x_indices, values


def _remove_rows(problem, reduced, rows, cols, keep):
    """Remap the kept sparse entries without slicing the full parameter tensor."""
    m_kept = np.count_nonzero(keep)
    positions = np.cumsum(keep, dtype=np.int64) - 1
    kept = np.flatnonzero(keep[rows])
    tensor_rows = positions[rows[kept]] + cols[kept] * m_kept
    entries = reduced[kept, :].tocoo()
    return sp.csc_array(
        (entries.data, (tensor_rows[entries.row], entries.col)),
        shape=(m_kept * (problem.x.size + 1), problem.A.shape[1]),
    )


class ExtractDirectCones(Reduction):
    """Move identity-pattern slack cones onto disjoint subvectors of x.

    Runs after ConeMatrixStuffing for solvers advertising DIR_CONE_KINDS.
    Detection uses the parameter tensor's structure, so extraction remains
    valid across DPP parameter updates. Cone order is CVXPY's standard
    order, with upper-triangle, sqrt(2)-scaled SvecPSD rows.
    """

    EXTRACTORS = {
        NonNeg: ('nonneg', _extract_nonneg),
        SOC: ('soc', _extract_soc),
        SvecPSD: ('psd_triangle', _extract_psd),
        ExpCone: ('exp', _extract_exp),
        PowCone3D: ('power', _extract_power),
        PowConeND: ('gen_power', _extract_gen_power),
    }

    def __init__(self, solver_context=None) -> None:
        kinds = solver_context.dir_cone_kinds if solver_context is not None else ()
        self._extractors = {cls: extractor for cls, (kind, extractor) in self.EXTRACTORS.items()
                            if kind in kinds}

    def accepts(self, problem) -> bool:
        return isinstance(problem, ParamConeProg) and bool(self._extractors)

    def apply(self, problem):
        if not self.accepts(problem):
            return problem, None
        # Interleave SOC / EXP / power rows into per-cone order before scanning.
        if not problem.formatted:
            problem = ConicSolver.format_constraints(problem, [0, 1, 2])
        if CONSTANT_ID not in problem.param_id_to_col:
            return problem, None

        problem.reduced_A.cache()
        reduced = problem.reduced_A.reduced_mat
        data_index = problem.reduced_A.problem_data_index
        if reduced is None or data_index is None:
            return problem, None
        reduced = reduced.tocsr()
        rows, indptr, _ = data_index
        cols = np.repeat(np.arange(problem.x.size + 1), np.diff(indptr))
        x_indices, values = _identity_rows(problem, reduced, rows, cols)

        dir_cones = list(problem.dir_cones)
        used = {idx for cone in dir_cones for idx in cone.indices}
        keep = np.ones(problem.constr_size, dtype=bool)
        constraints = []
        start = 0
        for constr in problem.constraints:
            stop = start + constr.size
            indices = x_indices[start:stop]
            extractor = self._extractors.get(type(constr))
            cones = []
            if extractor is not None and indices.size and np.all(indices >= 0):
                slots = set(indices.tolist())
                if len(slots) == indices.size and used.isdisjoint(slots):
                    cones = extractor(constr, indices, values[start:stop])
            if cones:
                dir_cones.extend(cones)
                used.update(slots)
                keep[start:stop] = False
            else:
                constraints.append(constr)
            start = stop
        if keep.all():
            return problem, None

        new_A = _remove_rows(problem, reduced, rows, cols, keep)
        return ParamConeProg(
            problem.q, problem.x, new_A, problem.variables, problem.var_id_to_col,
            constraints, problem.parameters, problem.param_id_to_col,
            P=problem.P, formatted=True,
            lower_bounds=problem.lower_bounds, upper_bounds=problem.upper_bounds,
            lb_tensor=problem.lb_tensor, ub_tensor=problem.ub_tensor,
            dir_cones=dir_cones,
        ), None

    def invert(self, solution, inverse_data):
        """The solver already restores direct-cone duals under their original IDs."""
        return solution
