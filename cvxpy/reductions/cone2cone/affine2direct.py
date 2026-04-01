"""
Copyright 2020 the CVXPY developers

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

from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.power import PowCone3D as PowCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.constraints.psd import SvecPSD as SvecPSD_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.zero import Zero as Zero_obj
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

FREE = 'fr'
ZERO = '0'
NONNEG = '+'
EXP = 'e'
DUAL_EXP = 'de'
SOC = 'q'
PSD = 's'
POW3D = 'pp3'
DUAL_POW3D = 'dp3'


class DualizeConeProg(Reduction):
    """Flags a ParamConeProg for dualized interpretation by the solver.

    This reduction sits between ConeMatrixStuffing and the solver.  It
    does **not** rewrite any matrices — it simply sets the ``dualized``
    flag on the ``ParamConeProg`` so that downstream solvers interpret
    the existing data ``(A, b, c, d, P, cone_dims)`` as the Lagrangian
    dual:

        (D)  max  -b'y  [- ½w'Pw]  + d
             s.t. A'y   [+ Pw]      = c
                  y in K*

    where ``K*`` is the dual of the cone ``K`` described by
    ``cone_dims``.  The slack variable ``w`` (size ``x.size``) is
    introduced by the solver only when ``P`` is not ``None``.

    Primal recovery
    ---------------
    ``x*`` is obtained from the dual of the equality constraint
    ``A'y = c`` (equivalently, ``A'y + Pw = c`` when ``P`` is
    present).  Original constraint duals are the optimal ``y``
    blocks, split by ``_split_primal_vars``.

    The solver is responsible for:
    1. Interpreting the flag (building a max problem, placing cones
       on variables, etc.).
    2. Returning a ``Solution`` with ``primal_vars = {'xx': ...,
       'barx': ...}`` and ``dual_vars = {s.EQ_DUAL: ...}``.

    Position in the chain
    ---------------------
    ::

        ... -> ConeMatrixStuffing -> DualizeConeProg -> Solver
    """

    def accepts(self, problem) -> bool:
        return (isinstance(problem, ParamConeProg)
                and not problem.is_mixed_integer()
                and not problem.dualized)

    def apply(self, problem):
        problem.dualized = True
        inv_data = {
            'x_id': problem.x.id,
            'constr_map': problem.constr_map,
            'K_dir': self._build_K_dir(problem.cone_dims),
        }
        return problem, inv_data

    def invert(self, solution, inverse_data):
        """Map a dualized solution back to the primal form."""
        status = solution.status
        prob_attr = solution.attr
        primal_vars, dual_vars = None, None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.opt_val
            primal_vars = {inverse_data['x_id']:
                           solution.dual_vars[s.EQ_DUAL]}
            dual_vars = dict()
            direct_prims = self._split_primal_vars(
                solution.primal_vars, inverse_data['K_dir'])
            constr_map = inverse_data['constr_map']
            i = 0
            for con in constr_map[Zero_obj]:
                dv = direct_prims[FREE][i:i + con.size]
                dual_vars[con.id] = dv if dv.size > 1 else dv.item()
                i += con.size
            i = 0
            for con in constr_map[NonNeg_obj]:
                dv = direct_prims[NONNEG][i:i + con.size]
                dual_vars[con.id] = dv if dv.size > 1 else dv.item()
                i += con.size
            i = 0
            for con in constr_map[SOC_obj]:
                block_len = con.shape[0]
                dv = np.concatenate(direct_prims[SOC][i:i + block_len])
                dual_vars[con.id] = dv
                i += block_len
            psd_cons = constr_map.get(PSD_obj, []) + constr_map.get(
                SvecPSD_obj, [])
            for i, con in enumerate(psd_cons):
                dv = direct_prims[PSD][i]
                dual_vars[con.id] = dv
            i = 0
            for con in constr_map[ExpCone_obj]:
                dv = direct_prims[DUAL_EXP][i:i + con.size]
                dual_vars[con.id] = dv
                i += con.size
            i = 0
            for con in constr_map[PowCone_obj]:
                dv = direct_prims[DUAL_POW3D][i:i + con.size]
                dual_vars[con.id] = dv
                i += con.size
        elif status == s.INFEASIBLE:
            status = s.UNBOUNDED
            opt_val = -np.inf
        elif status == s.INFEASIBLE_INACCURATE:
            status = s.UNBOUNDED_INACCURATE
            opt_val = -np.inf
        elif status == s.UNBOUNDED:
            status = s.INFEASIBLE
            opt_val = np.inf
        elif status == s.UNBOUNDED_INACCURATE:
            status = s.INFEASIBLE_INACCURATE
            opt_val = np.inf
        else:
            status = s.SOLVER_ERROR
            opt_val = np.nan
        return Solution(status, opt_val, primal_vars, dual_vars, prob_attr)

    @staticmethod
    def _build_K_dir(cone_dims):
        """Build the dual cone structure from primal ConeDims."""
        return {
            FREE: cone_dims.zero,
            NONNEG: cone_dims.nonneg,
            SOC: cone_dims.soc,
            PSD: cone_dims.psd,
            DUAL_EXP: cone_dims.exp,
            DUAL_POW3D: cone_dims.p3d,
        }

    @staticmethod
    def _split_primal_vars(primal_vars, K_dir):
        """Split a flat primal vector into per-cone blocks.

        ``primal_vars`` must contain ``'xx'`` (a flat vector from
        getxx) and optionally ``'barx'`` (a list of PSD vectors
        from getbarxj), which are split according to ``K_dir``.
        """
        xx = primal_vars['xx']
        result = {}
        idx = 0
        m_free = K_dir[FREE]
        if m_free > 0:
            result[FREE] = xx[idx:idx + m_free]
            idx += m_free
        m_pos = K_dir[NONNEG]
        if m_pos > 0:
            result[NONNEG] = xx[idx:idx + m_pos]
            idx += m_pos
        if K_dir[SOC]:
            soc_vars = []
            for dim in K_dir[SOC]:
                soc_vars.append(xx[idx:idx + dim])
                idx += dim
            result[SOC] = soc_vars
        if K_dir[DUAL_EXP]:
            n_exp = K_dir[DUAL_EXP]
            result[DUAL_EXP] = xx[idx:idx + 3 * n_exp]
            idx += 3 * n_exp
        if K_dir[DUAL_POW3D]:
            n_pow = len(K_dir[DUAL_POW3D])
            result[DUAL_POW3D] = xx[idx:idx + 3 * n_pow]
            idx += 3 * n_pow
        if 'barx' in primal_vars:
            result[PSD] = primal_vars['barx']
        return result
