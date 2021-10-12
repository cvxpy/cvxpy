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
import scipy as sp

from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone as ExpCone_obj
from cvxpy.constraints.nonpos import NonNeg as NonNeg_obj
from cvxpy.constraints.power import PowCone3D as PowCone_obj
from cvxpy.constraints.psd import PSD as PSD_obj
from cvxpy.constraints.second_order import SOC as SOC_obj
from cvxpy.constraints.zero import Zero as Zero_obj
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


class Dualize:
    """
    CVXPY represents cone programs as

        (P-Opt) min{ c.T @ x : A @ x + b in K } + d,

    where the corresponding dual is

        (D-Opt) max{ -b @ y : c = A.T @ y, y in K^* } + d.

    For some solvers, it is much easier to specify a problem of the form (D-Opt) than it
    is to specify a problem of the form (P-Opt). The purpose of this reduction is to handle
    mapping between (P-Opt) and (D-Opt) so that a solver interface can pretend the original
    problem was stated in terms (D-Opt).

    Usage
    -----
    Dualize applies to ParamConeProg problems. It accesses (P-Opt) data by calling
    ``c, d, A, b = problem.apply_parameters()``. It assumes the solver interface
    has already executed its ``format_constraints`` function on the ParamConeProg problem.

    A solver interface is responsible for calling both Dualize.apply and Dualize.invert.
    The call to Dualize.apply should be one of the first things that happens, and the
    call to Dualize.invert should be one of the last things that happens.

    The "data" dict returned by Dualize.apply is keyed by s.A, s.B, s.C, and 'K_dir',
    which respectively provide the dual constraint matrix (A.T), the dual constraint
    right-hand-side (c), the dual objective vector (-b), and the dual cones (K^*).
    The solver interface should interpret this data is a new primal problem, just with a
    maximization objective. Given a numerical solution, the solver interface should first
    construct a CVXPY Solution object where :math:`y` is a primal variable, divided into
    several blocks according to the structure of elementary cones appearing in K^*. The only
    dual variable we use is that corresponding to the equality constraint :math:`c = A^T y`.
    No attempt should be made to map unbounded / infeasible status codes for (D-Opt) back
    to unbounded / infeasible status codes for (P-Opt); all such mappings are handled in
    Dualize.invert. Refer to Dualize.invert for detailed documentation.

    Assumptions
    -----------
    The problem has no integer or boolean constraints. This is necessary because strong
    duality does not hold for problems with discrete constraints.

    Dualize.apply assumes "SOLVER.format_constraints()" has already been called. This
    assumption allows flexibility in how a solver interface chooses to vectorize a
    feasible set (e.g. how to order conic constraints, or how to vectorize the PSD cone).

    Additional notes
    ----------------

    Dualize.invert is written in a way which is agnostic to how a solver formats constraints,
    but it also imposes specific requirements on the input. Providing correct input to
    Dualize.invert requires consideration to the effect of ``SOLVER.format_constraints`` and
    the output of ``problem.apply_parameters``.
    """

    @staticmethod
    def apply(problem):
        c, d, A, b = problem.apply_parameters()
        Kp = problem.cone_dims  # zero, nonneg, exp, soc, psd
        Kd = {
            FREE: Kp.zero,  # length of block of unconstrained variables.
            NONNEG: Kp.nonneg,  # length of block of nonneg variables.
            SOC: Kp.soc,  # lengths of blocks of soc-constrained variables.
            PSD: Kp.psd,  # "orders" of PSD variables
            DUAL_EXP: Kp.exp,  # number of length-3 blocks of dual exp cone variables.
            DUAL_POW3D: Kp.p3d  # scale parameters for dual 3d power cones
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
            'constr_map': problem.constr_map,
            'x_id': problem.x.id,
            'K_dir': Kd,
            'dualized': True
        }
        return data, inv_data

    @staticmethod
    def invert(solution, inv_data):
        """
        ``solution`` is a CVXPY Solution object, formatted where

            (D-Opt) max{ -b @ y : c = A.T @ y, y in K^* } + d

        is the primal problem from the solver's perspective. The purpose of this function
        is to map such a solution back to the format

                (P-Opt) min{ c.T @ x : A @ x + b in K } + d.

        This function handles mapping of primal and dual variables, and solver status codes.
        The variable "x" in (P-Opt) is trivially populated from the dual variables to the
        constraint "c = A.T @ y" in (D-Opt). Status codes also map back in a simple way.

        Details on required formatting of solution.primal_vars
        ------------------------------------------------------

        We assume the dict solution.primal_vars is keyed by string-enums FREE ('fr'), NONNEG ('+'),
        SOC ('s'), PSD ('p'), and DUAL_EXP ('de'). The corresponding values are described below.

        solution.primal_vars[FREE] should be a single vector. It corresponds to the (possibly
        concatenated) components of "y" which are subject to no conic constraints. We map these
        variables back to dual variables for equality constraints in (P-Opt).

        solution.primal_vars[NONNEG] should also be a single vector, this time giving the
        possibly concatenated components of "y" which must be >= 0. We map these variables
        back to dual variables for inequality constraints in (P-Opt).

        solution.primal_vars[SOC] is a list of vectors specifying blocks of "y" which belong
        to the second-order-cone under the CVXPY standard ({ z : z[0] >= || z[1:] || }).
        We map these variables back to dual variables for SOC constraints in (P-Opt).

        solution.primal_vars[PSD] is a list of symmetric positive semidefinite matrices
        which result by lifting the vectorized PSD blocks of "y" back into matrix form.
        We assign these as dual variables to PSD constraints appearing in (P-Opt).

        solution.primal_vars[DUAL_EXP] is a vector of concatenated length-3 slices of y, where
        each constituent length-3 slice belongs to dual exponential cone as implied by the CVXPY
        standard of the primal exponential cone (see cvxpy/constraints/exponential.py:ExpCone).
        We map these back to dual variables for exponential cone constraints in (P-Opt).

        """
        status = solution.status
        prob_attr = solution.attr
        primal_vars, dual_vars = None, None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.opt_val + inv_data[s.OBJ_OFFSET]
            primal_vars = {inv_data['x_id']:
                           solution.dual_vars[s.EQ_DUAL]}
            dual_vars = dict()
            direct_prims = solution.primal_vars
            constr_map = inv_data['constr_map']
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
            for i, con in enumerate(constr_map[PSD_obj]):
                dv = direct_prims[PSD][i]
                dual_vars[con.id] = dv
            i = 0
            for con in constr_map[ExpCone_obj]:
                dv = direct_prims[DUAL_EXP][i:i + con.size]
                dual_vars[con.id] = dv
                i += con.size
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
            opt_val = np.NaN
        sol = Solution(status, opt_val, primal_vars, dual_vars, prob_attr)
        return sol


class Slacks:
    """
    CVXPY represents mixed-integer cone programs as

        (Aff)   min{ c.T @ x : A @ x + b in K,
                              x[bools] in {0, 1}, x[ints] in Z } + d.

    Some solvers do not accept input in the form (Aff). A general pattern we find
    across solver types is that the feasible set is represented by

        (Dir)   min{ f @ y : G @ y <=_{K_aff} h, y in K_dir
                             y[bools] in {0, 1}, y[ints] in Z } + d,

    where K_aff is built from a list convex cones which includes the zero cone (ZERO),
    and K_dir is built from a list of convex cones which includes the free cone (FREE).

    This reduction handles mapping back and forth between problems stated in terms
    of (Aff) and (Dir), by way of adding slack variables.

    Notes
    -----
    Support for semidefinite constraints has not yet been implemented in this
    reduction.

    If the problem has no integer constraints, then the Dualize reduction should be
    used instead.

    Because this reduction is only intended for mixed-integer problems, this reduction
    makes no attempt to recover dual variables when mapping between (Aff) and (Dir).
    """

    @staticmethod
    def apply(prob, affine):
        """
        "prob" is a ParamConeProg which represents

            (Aff)   min{ c.T @ x : A @ x + b in K,
                                  x[bools] in {0, 1}, x[ints] in Z } + d.

        We return data for an equivalent problem

            (Dir)   min{ f @ y : G @ y <=_{K_aff} h, y in K_dir
                                 y[bools] in {0, 1}, y[ints] in Z } + d,

        where

            (1) K_aff is built from cone types specified in "affine" (a list of strings),
            (2) a primal solution for (Dir) can be mapped back to a primal solution
                for (Aff) by selecting the leading ``c.size`` block of y's components.

        In the returned dict "data", data[s.A] = G, data[s.B] = h, data[s.C] = f,
        data['K_aff'] = K_aff, data['K_dir'] = K_dir, data[s.BOOL_IDX] = bools,
        and data[s.INT_IDX] = ints. The rows of G are ordered according to ZERO, then
        (as applicable) NONNEG, SOC, and EXP. If  "c" is the objective vector in (Aff),
        then ``y[:c.size]`` should contain the optimal solution to (Aff). The columns of
        G correspond first to variables in cones FREE, then NONNEG, then SOC, then EXP.
        The length of the free cone is equal to ``c.size``.

        Assumptions
        -----------
        The function call ``c, d, A, b = prob.apply_parameters()`` returns (A,b) with
        rows formatted first for the zero cone, then for the nonnegative orthant, then
        second order cones, then the exponential cone. Removing this assumption will
        require adding additional data to ParamConeProg objects.
        """
        c, d, A, b = prob.apply_parameters()  # A @ x + b in K
        A = -A  # A @ x <=_K b.
        cone_dims = prob.cone_dims
        if cone_dims.psd:
            # This will need to account for different conventions: does order-n
            # PSD constraint give rise to n**2 rows in A, or n*(n-1)//2 rows?
            raise NotImplementedError()

        for val in affine:
            if val not in {ZERO, NONNEG, EXP, SOC, POW3D}:
                raise NotImplementedError()
        if ZERO not in affine:
            affine.append(ZERO)

        cone_lens = {
            ZERO: cone_dims.zero,
            NONNEG: cone_dims.nonneg,
            SOC: sum(cone_dims.soc),
            EXP: 3 * cone_dims.exp,
            POW3D: 3 * len(cone_dims.p3d)
        }
        row_offsets = {
            ZERO: 0,
            NONNEG: cone_lens[ZERO],
            SOC: cone_lens[ZERO] + cone_lens[NONNEG],
            EXP: cone_lens[ZERO] + cone_lens[NONNEG] + cone_lens[SOC],
            POW3D: cone_lens[ZERO] + cone_lens[NONNEG] + cone_lens[SOC] + cone_lens[EXP]
        }
        # ^ If the rows of A are formatted in an order different from
        # zero -> nonneg -> soc -> exp -> pow, then the above block of code should
        # change. Right now there isn't enough data in (c, d, A, b, cone_dims,
        # constr_map) which allows us to figure out the ordering of these rows.
        A_aff, b_aff = [], []
        A_slk, b_slk = [], []
        total_slack = 0
        for co_type in [ZERO, NONNEG, SOC, EXP, POW3D]:
            # ^ The order of that list means that the matrix "G" in "G @ z <=_{K_aff} h"
            # will always have rows ordered by the zero cone, then the nonnegative orthant,
            # then second order cones, and finally exponential cones. Changing the order
            # of items in this list would change the order of row blocks in "G".
            #
            # If the order is changed, then this affects which columns of the final matrix
            # "G" correspond to which types of cones. For example, [ZERO, SOC, EXP, POW3D, NONNEG]
            # and NONNEG is not in "affine", then the columns of G with nonnegative variables
            # occur after all free variables, soc variables, exp variables, and pow3d variables.
            co_dim = cone_lens[co_type]
            if co_dim > 0:
                r = row_offsets[co_type]
                A_temp = A[r:r + co_dim, :]
                b_temp = b[r:r + co_dim]
                if co_type in affine:
                    A_aff.append(A_temp)
                    b_aff.append(b_temp)
                else:
                    total_slack += b_temp.size
                    A_slk.append(A_temp)
                    b_slk.append(b_temp)
        K_dir = {
            FREE: prob.x.size,
            NONNEG: 0 if NONNEG in affine else cone_dims.nonneg,
            SOC: [] if SOC in affine else cone_dims.soc,
            EXP: 0 if EXP in affine else cone_dims.exp,
            PSD: [],  # not currently supported in this reduction
            DUAL_EXP: 0,  # not currently supported in cvxpy
            POW3D: [] if POW3D in affine else cone_dims.p3d,
            DUAL_POW3D: []  # not currently supported in cvxpy
        }
        K_aff = {
            NONNEG: cone_dims.nonneg if NONNEG in affine else 0,
            SOC: cone_dims.soc if SOC in affine else [],
            EXP: cone_dims.exp if EXP in affine else 0,
            PSD: [],  # currently not supported in this reduction
            ZERO: cone_dims.zero + total_slack,
            POW3D: cone_dims.p3d if POW3D in affine else []
        }

        data = dict()
        if A_slk:
            # We need to introduce slack variables.
            A_slk = sp.sparse.vstack(tuple(A_slk))
            eye = sp.sparse.eye(total_slack)
            if A_aff:
                A_aff = sp.sparse.vstack(tuple(A_aff), format='csr')
                G = sp.sparse.bmat([[A_slk, eye], [A_aff, None]])
                h = np.concatenate(b_slk + b_aff)  # concatenate lists, then turn to vector
            else:
                G = sp.sparse.hstack((A_slk, eye))
                h = np.concatenate(b_slk)
            f = np.concatenate((c, np.zeros(total_slack)))
        elif A_aff:
            # No slack variables were introduced.
            G = sp.sparse.vstack(tuple(A_aff), format='csr')
            h = np.concatenate(b_aff)
            f = c
        else:
            raise ValueError()

        data[s.A] = G
        data[s.B] = h
        data[s.C] = f
        data[s.BOOL_IDX] = [int(t[0]) for t in prob.x.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in prob.x.integer_idx]
        data['K_dir'] = K_dir
        data['K_aff'] = K_aff

        inv_data = dict()
        inv_data['x_id'] = prob.x.id
        inv_data['K_dir'] = K_dir
        inv_data['K_aff'] = K_aff
        inv_data[s.OBJ_OFFSET] = d

        return data, inv_data

    @staticmethod
    def invert(solution, inv_data):
        if solution.status in s.SOLUTION_PRESENT:
            prim_vars = solution.primal_vars
            x = prim_vars[FREE]
            del prim_vars[FREE]
            prim_vars[inv_data['x_id']] = x
        solution.opt_val += inv_data[s.OBJ_OFFSET]
        return solution
