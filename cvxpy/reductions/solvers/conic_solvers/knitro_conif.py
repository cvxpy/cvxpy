"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver, dims_to_solver_dict
from cvxpy.utilities.citations import CITATION_DICT


def kn_isinf(x) -> bool:
    """Check if x is -inf or inf."""
    if x <= -np.inf or x >= np.inf:
        return True
    if x <= float("-inf") or x >= float("inf"):
        return True

    import knitro as kn

    if x <= -kn.KN_INFINITY or x >= kn.KN_INFINITY:
        return True
    return False


def kn_rm_inf(a) -> tuple[list[int], list[float]]:
    """Convert -inf to -kn.KN_INFINITY and inf to kn.KN_INFINITY."""
    i, vs = [], []
    for j, v in enumerate(a):
        if not kn_isinf(v):
            i.append(j)
            vs.append(v)
    return i, vs


class Dims:
    def __init__(self, dims: dict):
        self.n_eq_cons = int(dims.get(s.EQ_DIM, 0))
        self.n_ineq_cons = int(dims.get(s.LEQ_DIM, 0))
        self.soc_dims = [int(d) for d in dims.get(s.SOC_DIM, [])]
        self.n_exps = int(dims.get(s.EXP_DIM, 0))
        self.p3d_exps = dims.get("p", [])
        self.psd_dims = dims.get(s.PSD_DIM, [])
        self.n_p3ds = len(self.p3d_exps)
        self.n_socs = len(self.soc_dims)
        self.n_psds = len(self.psd_dims)
        self.n_exp_cons = self.n_exps
        self.n_p3d_cons = self.n_p3ds
        self.n_soc_cons = self.n_socs
        self.n_psd_cons = sum(d**2 for d in self.psd_dims)
        self.n_cone_cons = self.n_soc_cons + self.n_exp_cons + self.n_psd_cons + self.n_p3d_cons
        self.n_soc_vars = sum(self.soc_dims)
        self.n_exp_vars = 3 * self.n_exp_cons
        self.n_psd_vars = sum(d**2 for d in self.psd_dims)
        self.n_p3d_vars = 3 * self.n_p3d_cons
        self.n_cone_vars = self.n_soc_vars + self.n_exp_vars + self.n_psd_vars + self.n_p3d_vars


class CB:
    # Knitro callback
    def __init__(self, f, grad=None, hess=None):
        self.f = f
        self.grad = grad
        self.hess = hess


class Ctx:
    def __init__(self, n: int, vp: int, cp: int, a=None):
        self.n = n  # Number of constraints
        self.vp = vp  # Variable position
        self.cp = cp  # Constraint position
        if a is not None:
            self.a = np.array(a)


def build_exp_cb() -> CB:
    import knitro as kn

    def f(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALFC:
            return -1
        n = ctx.n
        vp = ctx.vp
        v = np.asarray(req.x[vp : vp + 3 * n])
        x = v[0::3]
        y = v[1::3]
        z = v[2::3]
        xy = np.divide(x, y)
        with np.errstate(over="ignore"):
            exp = np.exp(xy)
        mask = (np.isclose(y, 0.0)) | np.isnan(xy) | np.isnan(exp) | np.isinf(exp)
        ind = np.where(x <= 0.0, -1.0, 0.0)
        res.c[:n] = np.where(mask, ind, y * exp - z)
        return 0

    def grad(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALGA:
            return -1
        n = ctx.n
        vp = ctx.vp
        v = np.asarray(req.x[vp : vp + 3 * n])
        x = v[0::3]
        y = v[1::3]
        xy = np.divide(x, y)
        with np.errstate(over="ignore"):
            exp = np.exp(xy)
        mask = (np.isclose(y, 0.0)) | np.isnan(xy) | np.isnan(exp) | np.isinf(exp)
        res.jac[0::3] = np.where(mask, np.inf, exp)
        res.jac[1::3] = np.where(mask, np.inf, (1 - xy) * exp)
        res.jac[2::3] = np.where(mask, np.inf, -1.0)
        return 0

    def hess(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALH and req.type != kn.KN_RC_EVALH_NO_F:
            return -1
        n = ctx.n
        vp = ctx.vp
        cp = ctx.cp
        v = np.asarray(req.x[vp : vp + 3 * n])
        u = np.asarray(req.lambda_[cp : cp + n])
        x = v[0::3]
        y = v[1::3]
        xy = np.divide(x, y)
        with np.errstate(over="ignore"):
            exp = np.exp(xy)
        mask = (np.isclose(y, 0.0)) | np.isnan(xy) | np.isnan(exp) | np.isinf(exp)
        iy = np.divide(1, y)
        hx = iy * exp * u
        res.hess[0::3] = np.where(mask, np.inf, hx)
        res.hess[1::3] = np.where(mask, np.inf, -xy * hx)
        res.hess[2::3] = np.where(mask, np.inf, (xy**2) * hx)
        return 0

    return CB(f=f, grad=grad, hess=hess)


def build_pow3d_cb() -> CB:
    import knitro as kn

    def f(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALFC:
            return -1
        n = ctx.n
        vp = ctx.vp
        a = ctx.a
        v = np.asarray(req.x[vp : vp + 3 * n])
        x = v[0::3]
        y = v[1::3]
        z = v[2::3]
        res.c[:n] = -np.power(x, a) * np.power(y, 1 - a) + np.abs(z)
        return 0

    def grad(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALGA:
            return -1
        n = ctx.n
        vp = ctx.vp
        a = ctx.a
        v = np.asarray(req.x[vp : vp + 3 * n])
        x = v[0::3]
        y = v[1::3]
        z = v[2::3]
        res.jac[0::3] = -a * np.power(x, a - 1) * np.power(y, 1 - a)
        res.jac[1::3] = -(1 - a) * np.power(x, a) * np.power(y, -a)
        res.jac[2::3] = np.sign(z)
        return 0

    def hess(_, cb: kn.CB_context, req: kn.KN_eval_request, res: kn.KN_eval_result, ctx: Ctx):
        if req.type != kn.KN_RC_EVALH and req.type != kn.KN_RC_EVALH_NO_F:
            return -1
        n = ctx.n
        vp = ctx.vp
        cp = ctx.cp
        a = ctx.a
        b = a * (1 - a)
        v = np.asarray(req.x[vp : vp + 3 * n])
        u = np.asarray(req.lambda_[cp : cp + n])
        x = v[0::3]
        y = v[1::3]
        res.hess[0::3] = b * np.power(x, a - 2) * np.power(y, 1 - a) * u
        res.hess[1::3] = -b * np.power(x, a - 1) * np.power(y, -a) * u
        res.hess[2::3] = b * np.power(x, a) * np.power(y, -a - 1) * u
        return 0

    return CB(f=f, grad=grad, hess=hess)


class KNITRO(ConicSolver):
    """
    Conic interface for the Knitro solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D, PSD]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"
    N_VARS_KEY = "n_vars"
    N_CONS_KEY = "n_cons"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY, Y_INIT_KEY]

    EXP_CONE_ORDER = [0, 1, 2]
    EXP_DOUBLE_LIMIT = 705.0

    # Map of Knitro status to CVXPY status.
    STATUS_MAP = {
        0: s.OPTIMAL,
        -100: s.OPTIMAL_INACCURATE,
        -101: s.USER_LIMIT,
        -102: s.USER_LIMIT,
        -103: s.USER_LIMIT,
        -200: s.INFEASIBLE,
        -201: s.INFEASIBLE,
        -202: s.INFEASIBLE,
        -203: s.INFEASIBLE,
        -204: s.INFEASIBLE,
        -205: s.INFEASIBLE,
        -300: s.UNBOUNDED,
        -301: s.UNBOUNDED,
        -400: s.USER_LIMIT,
        -401: s.USER_LIMIT,
        -402: s.USER_LIMIT,
        -403: s.USER_LIMIT,
        -404: s.USER_LIMIT,
        -405: s.USER_LIMIT,
        -406: s.USER_LIMIT,
        -410: s.USER_LIMIT,
        -411: s.USER_LIMIT,
        -412: s.USER_LIMIT,
        -413: s.USER_LIMIT,
        -415: s.USER_LIMIT,
        -416: s.USER_LIMIT,
        -500: s.SOLVER_ERROR,
        -501: s.SOLVER_ERROR,
        -502: s.SOLVER_ERROR,
        -503: s.SOLVER_ERROR,
        -504: s.SOLVER_ERROR,
        -505: s.SOLVER_ERROR,
        -506: s.SOLVER_ERROR,
        -507: s.SOLVER_ERROR,
        -508: s.SOLVER_ERROR,
        -509: s.SOLVER_ERROR,
        -510: s.SOLVER_ERROR,
        -511: s.SOLVER_ERROR,
        -512: s.SOLVER_ERROR,
        -513: s.SOLVER_ERROR,
        -514: s.SOLVER_ERROR,
        -515: s.SOLVER_ERROR,
        -516: s.SOLVER_ERROR,
        -517: s.SOLVER_ERROR,
        -518: s.SOLVER_ERROR,
        -519: s.SOLVER_ERROR,
        -520: s.SOLVER_ERROR,
        -521: s.SOLVER_ERROR,
        -522: s.SOLVER_ERROR,
        -523: s.SOLVER_ERROR,
        -524: s.SOLVER_ERROR,
        -525: s.SOLVER_ERROR,
        -526: s.SOLVER_ERROR,
        -527: s.SOLVER_ERROR,
        -528: s.SOLVER_ERROR,
        -529: s.SOLVER_ERROR,
        -530: s.SOLVER_ERROR,
        -531: s.SOLVER_ERROR,
        -532: s.SOLVER_ERROR,
        -600: s.SOLVER_ERROR,
    }  # MEM_LIMIT

    def name(self):
        """The name of the solver."""
        return s.KNITRO

    def import_solver(self) -> None:
        """Imports the solver."""
        import knitro

        knitro

    def supports_quad_obj(self):
        return True

    def accepts(self, problem) -> bool:
        return super(KNITRO, self).accepts(problem)

    def apply(self, problem: ParamConeProg):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(KNITRO, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data["is_mip"] = data[s.BOOL_IDX] or data[s.INT_IDX]
        return data, inv_data

    def invert(self, results, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        import knitro as kn

        if KNITRO.CONTEXT_KEY not in results:
            return failure_solution(s.SOLVER_ERROR)

        kc = results[KNITRO.CONTEXT_KEY]
        num_iters = kn.KN_get_number_iters(kc)
        solve_time = kn.KN_get_solve_time_real(kc)
        attr = {
            s.SOLVE_TIME: solve_time,
            s.NUM_ITERS: num_iters,
            s.EXTRA_STATS: kc,
        }

        if s.STATUS in results and results[s.STATUS] == s.SOLVER_ERROR:
            solution = failure_solution(s.SOLVER_ERROR, attr)
        else:
            status_kn, obj_kn, x_kn, y_kn = kn.KN_get_solution(kc)
            status = self.STATUS_MAP.get(status_kn, s.SOLVER_ERROR)

            if status == s.UNBOUNDED:
                solution = Solution(status, -np.inf, {}, {}, attr)
            elif (status not in s.SOLUTION_PRESENT) or (x_kn is None):
                solution = failure_solution(status, attr)
            else:
                n_vars = int(results[KNITRO.N_VARS_KEY])
                n_cons = int(results[KNITRO.N_CONS_KEY])

                obj = obj_kn + inverse_data[s.OFFSET]
                x_kn = x_kn[:n_vars]
                x = np.array(x_kn)
                primal_vars = {inverse_data[KNITRO.VAR_ID]: x}

                dual_vars = None
                is_mip = bool(inverse_data.get("is_mip", False))
                y_kn = kn.KN_get_con_dual_values(kc)
                if y_kn is not None and not is_mip:
                    dims = dims_to_solver_dict(inverse_data[s.DIMS] or {})
                    y_kn = y_kn[:n_cons]
                    n_eqs = int(dims.get(s.EQ_DIM, 0))
                    y = np.array(y_kn)
                    eq_dual_vars = utilities.get_dual_values(
                        y[:n_eqs],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.EQ_CONSTR],
                    )
                    ineq_dual_vars = utilities.get_dual_values(
                        y[n_eqs:],
                        utilities.extract_dual_value,
                        inverse_data[KNITRO.NEQ_CONSTR],
                    )
                    dual_vars = {**eq_dual_vars, **ineq_dual_vars}
                solution = Solution(status, obj, primal_vars, dual_vars, attr)
        # Free the Knitro context.
        kn.KN_free(kc)
        return solution

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        The result of the call to the knitro solver.
        """
        import knitro as kn

        P = data.get(s.P)
        c = data.get(s.C)
        b = data.get(s.B)
        A = data.get(s.A)
        lb = data.get(s.LOWER_BOUNDS)
        ub = data.get(s.UPPER_BOUNDS)
        dims = Dims(dims_to_solver_dict(data.get(s.DIMS)))

        results = {}
        try:
            kc = kn.KN_new()
        except Exception:
            results[s.STATUS] = s.SOLVER_ERROR
            return results

        results[KNITRO.CONTEXT_KEY] = kc

        if not verbose:
            # Disable Knitro output.
            kn.KN_set_int_param(kc, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_NONE)

        n_vars = int(c.shape[0])
        results[KNITRO.N_VARS_KEY] = n_vars

        # Add n variables to the problem.
        kn.KN_add_vars(kc, n_vars)

        # Set the lower and upper bounds on the variables.
        if lb is not None:
            i, lb = kn_rm_inf(lb)
            kn.KN_set_var_lobnds(kc, indexVars=i, xLoBnds=lb)
        if ub is not None:
            i, ub = kn_rm_inf(ub)
            kn.KN_set_var_upbnds(kc, indexVars=i, xUpBnds=ub)

        # Set the variable types.
        vts = [kn.KN_VARTYPE_CONTINUOUS] * n_vars
        if s.BOOL_IDX in data:
            for j in data[s.BOOL_IDX]:
                vts[j] = kn.KN_VARTYPE_BINARY
        if s.INT_IDX in data:
            for j in data[s.INT_IDX]:
                vts[j] = kn.KN_VARTYPE_INTEGER
        kn.KN_set_var_types(kc, xTypes=vts)

        # Set the initial values of the primal variables.
        if KNITRO.X_INIT_KEY in solver_opts:
            i, vs = solver_opts[KNITRO.X_INIT_KEY]
            kn.KN_set_var_primal_init_values(kc, indexVars=i, xInitVals=vs)

        # Add constraints to the problem.
        n_cons = int(A.shape[0]) if A is not None else 0
        results[KNITRO.N_CONS_KEY] = n_cons
        if n_cons > 0:
            kn.KN_add_cons(kc, n_cons)

        if dims.n_cone_vars > 0:
            kn.KN_add_vars(kc, dims.n_cone_vars)
        if dims.n_cone_cons > 0:
            kn.KN_add_cons(kc, dims.n_cone_cons)
        if dims.n_psds > 0:
            kn.KN_add_vars(kc, dims.n_psd_vars)

        D = sp.coo_matrix(A)
        if D.nnz != 0:
            cis, vis, coefs = D.row, D.col, D.data
            kn.KN_add_con_linear_struct(kc, indexCons=cis, indexVars=vis, coefs=coefs)

        vp, cp = 0, 0
        if dims.n_eq_cons > 0:
            cis = cp + np.arange(dims.n_eq_cons)
            kn.KN_set_con_eqbnds(kc, indexCons=cis, cEqBnds=b[cis])
        cp += dims.n_eq_cons

        if dims.n_ineq_cons > 0:
            cis = cp + np.arange(dims.n_ineq_cons)
            kn.KN_set_con_upbnds(kc, indexCons=cis, cUpBnds=b[cis])
        cp += dims.n_ineq_cons
        vp += n_vars

        if dims.n_cone_vars > 0:
            vis = vp + np.arange(dims.n_cone_vars)
            cis = cp + np.arange(dims.n_cone_vars)
            coefs = np.ones_like(vis)
            kn.KN_set_con_eqbnds(kc, indexCons=cis, cEqBnds=b[cis])
            kn.KN_add_con_linear_struct(kc, indexCons=cis, indexVars=vis, coefs=coefs)
        cp += dims.n_cone_vars

        if dims.n_socs > 0:
            cis = cp + np.arange(dims.n_soc_cons)
            vis = vp + np.insert(np.cumsum(dims.soc_dims), 0, 0)[:-1]
            bnds = np.zeros_like(cis)
            coefs = -np.ones_like(vis)
            kn.KN_set_con_upbnds(kc, indexCons=cis, cUpBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=vis, xLoBnds=bnds)
            kn.KN_add_con_quadratic_struct(
                kc, indexCons=cis, indexVars1=vis, indexVars2=vis, coefs=coefs
            )

        for k in range(dims.n_soc_cons):
            d = dims.soc_dims[k]
            vis = vp + np.arange(1, d)
            coefs = np.ones_like(vis)
            kn.KN_add_con_quadratic_struct(
                kc, indexCons=cp + k, indexVars1=vis, indexVars2=vis, coefs=coefs
            )
            vp += d
        cp += dims.n_soc_cons

        if dims.n_exps > 0:
            cis = cp + np.arange(dims.n_exp_cons)
            vis = vp + np.arange(dims.n_exp_vars)
            bnds = np.zeros_like(cis)
            kn.KN_set_con_upbnds(kc, indexCons=cis, cUpBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=vis[1::3], xLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=vis[2::3], xLoBnds=bnds)

            cb = build_exp_cb()

            kb = kn.KN_add_eval_callback(kc, indexCons=cis, funcCallback=cb.f)
            jcis = np.repeat(cis, 3)
            jvis = vis
            kn.KN_set_cb_grad(kc, kb, jacIndexCons=jcis, jacIndexVars=jvis, gradCallback=cb.grad)
            hvis = np.repeat(vis[0::3], 3)
            hvis1 = hvis + np.tile(np.array([0, 0, 1]), dims.n_exps)
            hvis2 = hvis + np.tile(np.array([0, 1, 1]), dims.n_exps)
            kn.KN_set_cb_hess(
                kc, kb, hessIndexVars1=hvis1, hessIndexVars2=hvis2, hessCallback=cb.hess
            )
            ctx = Ctx(n=dims.n_exps, vp=vp, cp=cp)
            kn.KN_set_cb_user_params(kc, kb, ctx)
        cp += dims.n_exp_cons
        vp += dims.n_exp_vars

        if dims.n_psds > 0:
            cis = cp + np.arange(dims.n_psd_cons)
            vis = vp + np.arange(dims.n_psd_vars)
            bnds = np.zeros_like(cis)
            coefs = -np.ones_like(vis)
            kn.KN_set_con_eqbnds(kc, indexCons=cis, cEqBnds=bnds)
            kn.KN_add_con_linear_struct(kc, indexCons=cis, indexVars=vis, coefs=coefs)
            vp += dims.n_psd_vars

        vp += dims.n_p3d_vars
        for k in range(dims.n_psds):
            d = dims.psd_dims[k]
            vis = vp + np.arange(d**2)
            cis = cp + np.arange(d**2)
            for i in range(d):
                for j in range(d):
                    vis1 = vis[d * i : d * (i + 1)]
                    vis2 = vis[d * j : d * (j + 1)]
                    coefs = np.ones_like(vis1)
                    kn.KN_add_con_quadratic_struct(
                        kc, indexCons=cis[d * i + j], indexVars1=vis1, indexVars2=vis2, coefs=coefs
                    )
            cp += d**2
            vp += d**2
        vp -= dims.n_p3d_vars

        if dims.n_p3ds > 0:
            cis = cp + np.arange(dims.n_p3d_cons)
            vis = vp + np.arange(dims.n_p3d_vars)
            bnds = np.zeros_like(cis)
            kn.KN_set_con_upbnds(kc, indexCons=cis, cUpBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=vis[0::3], xLoBnds=bnds)
            kn.KN_set_var_lobnds(kc, indexVars=vis[1::3], xLoBnds=bnds)

            cb = build_pow3d_cb()

            kb = kn.KN_add_eval_callback(kc, indexCons=cis, funcCallback=cb.f)
            jcis = np.repeat(cis, 3)
            jvis = vis
            kn.KN_set_cb_grad(kc, kb, jacIndexCons=jcis, jacIndexVars=jvis, gradCallback=cb.grad)
            hvis = np.repeat(vis[0::3], 3)
            hvis1 = hvis + np.tile(np.array([0, 0, 1]), dims.n_p3ds)
            hvis2 = hvis + np.tile(np.array([0, 1, 1]), dims.n_p3ds)
            kn.KN_set_cb_hess(
                kc, kb, hessIndexVars1=hvis1, hessIndexVars2=hvis2, hessCallback=cb.hess
            )
            ctx = Ctx(n=dims.n_p3ds, vp=vp, cp=cp, a=dims.p3d_exps)
            kn.KN_set_cb_user_params(kc, kb, ctx)
        cp += dims.n_p3d_cons
        vp += dims.n_p3d_vars

        # Set the initial values of the dual variables.
        if KNITRO.Y_INIT_KEY in solver_opts:
            i, vs = solver_opts[KNITRO.Y_INIT_KEY]
            kn.KN_set_con_dual_init_values(kc, indexCons=i, yInitVals=vs)

        # Set the linear part of the objective function.
        if c is not None:
            vis = np.arange(n_vars)
            kn.KN_add_obj_linear_struct(kc, indexVars=vis, coefs=c)

        # Set the quadratic part of the objective function.
        if P is not None and P.nnz != 0:
            Q = sp.coo_matrix(0.5 * P)
            vis1, vis2, coefs = Q.row, Q.col, Q.data
            kn.KN_add_obj_quadratic_struct(kc, indexVars1=vis1, indexVars2=vis2, coefs=coefs)

        # Set the sense of the objective function.
        kn.KN_set_obj_goal(kc, kn.KN_OBJGOAL_MINIMIZE)

        # Set the values of the parameters.
        for k, v in solver_opts.items():
            if k in KNITRO.INTERFACE_ARGS:
                continue
            pid = kn.KN_get_param_id(kc, k)
            pt = kn.KN_get_param_type(kc, pid)
            fn = kn.KN_set_char_param
            if pt == kn.KN_PARAMTYPE_INTEGER:
                fn = kn.KN_set_int_param
            elif pt == kn.KN_PARAMTYPE_FLOAT:
                fn = kn.KN_set_double_param
            fn(kc, pid, v)

        # Optimize the problem.
        try:
            kn.KN_solve(kc)
        except Exception:  # Error in the solution
            results[s.STATUS] = s.SOLVER_ERROR

        # Cache the Knitro context.
        if solver_cache is not None:
            solver_cache[self.name()] = kc

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT[s.KNITRO]
