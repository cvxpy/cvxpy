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
from cvxpy import settings as s
from cvxpy.reductions.solution import Solution
from cvxpy.constraints.zero import Zero as Zero_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims
import numpy as np
import scipy as sp


FREE = 'fr'
ZERO = '0'
NONNEG = '+'
EXP = 'e'
DUAL_EXP = 'de'
SOC = 's'
PSD = 'p'
CONSTR_MAP = 'constr_map'


class Dualize(object):

    @staticmethod
    def apply(problem):
        c, d, A, b = problem.apply_parameters()
        """
        min{ c.T @ x + d : A @ x + b in K } == max{ -b @ y : c = A.T @ y, y in K^* } + d
        """
        Kp = problem.cone_dims  # zero, nonneg, exp, soc, psd
        Kd = {
            FREE: Kp.zero,  # length of block of unconstrained variables.
            NONNEG: Kp.nonneg,  # length of block of nonneg variables.
            SOC: Kp.soc,  # lengths of blocks of soc-constrained variables.
            PSD: Kp.psd,  # "orders" of PSD variables
            DUAL_EXP: Kp.exp  # number of length-3 blocks of dual exp cone variables.
        }
        data = {
            s.A: A.T,
            s.B: c,
            s.C: -b,
            'K_dir': Kd,
            'dualized': True
        }
        inv_data = {
            s.OBJ_OFFSET: d,
            CONSTR_MAP: problem.constr_map,
            'x_id': problem.x.id,
            'K_dir': Kd
        }
        return data, inv_data

    @staticmethod
    def _invert_vars(solution, inv_data):
        primal_vars = {inv_data['x_id']:
                       solution.dual_vars[s.EQ_DUAL]}
        dual_vars = dict()
        direct_prims = solution.primal_vars
        constr_map = inv_data[CONSTR_MAP]
        i = 0
        for con in constr_map[Zero_obj]:
            dv = direct_prims[FREE][i:i + con.size]
            dual_vars[con.id] = dv
            i += con.size
        i = 0
        for con in constr_map[NonNeg_obj]:
            dv = direct_prims[NONNEG][i:i + con.size]
            dual_vars[con.id] = dv
            i += con.size
        i = 0
        for con in constr_map[SOC_obj]:
            block_len = con.shape[0]
            dv = np.concatenate(direct_prims[SOC][i:i + block_len])
            dual_vars[con.id] = dv
            i += block_len
        if len(constr_map[PSD_obj]) > 0:
            raise NotImplementedError()
        i = 0
        for con in constr_map[ExpCone_obj]:
            dv = direct_prims[DUAL_EXP][i:i + con.size]
            dual_vars[con.id] = dv
            i += con.size
        return primal_vars, dual_vars

    @staticmethod
    def invert(solution, inv_data):
        status = solution.status
        prob_attr = solution.attr
        primal_vars, dual_vars = None, None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.opt_val + inv_data[s.OBJ_OFFSET]
            primal_vars, dual_vars = Dualize._invert_vars(solution, inv_data)
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
            status = s.UNKNOWN
            opt_val = np.NaN
        sol = Solution(status, opt_val, primal_vars, dual_vars, prob_attr)
        return sol


class Slacks(object):

    @staticmethod
    def apply(problem, affine):
        c, d, A, b = problem.apply_parameters()  # A @ x + b in K
        A = -A  # A @ x <=_K b.
        cone_dims = problem.cone_dims
        if cone_dims.psd:
            # This will need to account for different conventions: does order-n
            # PSD constraint give rise to n**2 rows in A, or n*(n-1)//2 rows?
            raise NotImplementedError()

        for val in affine:
            if val not in {ZERO, NONNEG, EXP, SOC}:
                raise ValueError()
        if ZERO not in affine:
            affine.append(ZERO)

        cone_lens = {
            ZERO: cone_dims.zero,
            NONNEG: cone_dims.nonneg,
            SOC: sum(cone_dims.soc),
            EXP: 3 * cone_dims.exp
        }
        row_offsets = {
            ZERO: 0,
            NONNEG: cone_lens[ZERO],
            SOC: cone_lens[ZERO] + cone_lens[NONNEG],
            EXP: cone_lens[ZERO] + cone_lens[NONNEG] + cone_lens[SOC]
        }
        A_aff, b_aff = [], []
        A_slk, b_slk = [], []
        total_slack = 0
        for co_type in [ZERO, NONNEG, SOC, EXP]:
            co_dim = cone_lens[co_type]
            if co_dim > 0:
                r = row_offsets[co_type]
                A_temp = A[r:r + co_dim]
                b_temp = b[r:r + co_dim]
                if co_type in affine:
                    A_aff.append(A_temp)
                    b_aff.append(b_temp)
                else:
                    total_slack += b_temp.size
                    A_slk.append(A_temp)
                    b_slk.append(b_temp)
        K_dir = {
            FREE: problem.x.size,
            NONNEG: 0 if NONNEG in affine else cone_dims.nonneg,
            SOC: [] if SOC in affine else cone_dims.soc,
            EXP: 0 if EXP in affine else cone_dims.exp,
            PSD: [],
            DUAL_EXP: 0
        }
        K_aff = {
            NONNEG: cone_dims.nonneg if NONNEG in affine else 0,
            SOC: cone_dims.soc if SOC in affine else [],
            EXP: cone_dims.exp if EXP in affine else 0,
            PSD: [],
            ZERO: cone_dims.zero + total_slack
        }

        data = dict()
        if A_slk:
            A_slk = sp.sparse.vstack(tuple(A_slk))
            eye = sp.sparse.eye(total_slack)
            if A_aff:
                A_aff = sp.sparse.vstack(tuple(A_aff), format='csr')
                A = sp.sparse.bmat([[A_slk, eye], [A_aff, None]])
                b = np.concatenate(b_slk + b_aff)
            else:
                A = sp.sparse.hstack((A_slk, eye))
                b = np.concatenate(b_slk)
            c = np.concatenate((c, np.zeros(total_slack)))
        elif A_aff:
            A = sp.sparse.vstack(tuple(A_aff), format='csr')
            b = np.concatenate(b_aff)
        else:
            raise ValueError()

        data[s.A] = A
        data[s.B] = b
        data[s.C] = c
        data[s.BOOL_IDX] = [int(t[0]) for t in problem.x.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in problem.x.integer_idx]
        data['K_dir'] = K_dir
        data['K_aff'] = K_aff

        inv_data = dict()
        inv_data['x_id'] = problem.x.id
        inv_data['is_LP'] = (cone_lens[SOC] + cone_lens[EXP]) == 0
        inv_data['K_dir'] = K_dir
        inv_data['K_aff'] = K_aff
        inv_data['integer_variables'] = data[s.BOOL_IDX] or data[s.INT_IDX]
        inv_data[s.OBJ_OFFSET] = d

        return data, inv_data

    @staticmethod
    def invert(solution, inv_data):
        prim_vars = solution.primal_vars
        x = prim_vars[FREE]
        del prim_vars[FREE]
        prim_vars[inv_data['x_id']] = x
        solution.opt_val += inv_data[s.OBJ_OFFSET]
        return solution
