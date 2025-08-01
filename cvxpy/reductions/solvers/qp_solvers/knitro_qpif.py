"""
Copyright 2025 CVXPY developers

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
from scipy import sparse

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
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


class KNITRO(QpSolver):
    """QP interface for the Knitro solver"""

    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True

    # Keys:
    CONTEXT_KEY = "context"
    X_INIT_KEY = "x_init"
    Y_INIT_KEY = "y_init"

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = [X_INIT_KEY, Y_INIT_KEY]

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
        return s.KNITRO

    def import_solver(self) -> None:
        pass

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        return super(KNITRO, self).apply(problem)

    def invert(self, results, inverse_data):
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
                obj = obj_kn + inverse_data[s.OFFSET]
                x = np.array(x_kn)
                primal_vars = {KNITRO.VAR_ID: x}
                dual_vars = None
                is_mip = bool(inverse_data.get("is_mip", False))
                if y_kn is not None and not is_mip:
                    y = np.array(y_kn)
                    dual_vars = {KNITRO.DUAL_VAR_ID: y}
                solution = Solution(status, obj, primal_vars, dual_vars, attr)

        # Free the Knitro context.
        kn.KN_free(kc)
        return solution

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

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
        solver_cache: None
            None

        Returns
        -------
        The result of the call to the knitro solver.
        """
        import knitro as kn

        P = data[s.P]
        q = data[s.Q]
        A = data[s.A].tocoo()
        b = data[s.B]
        F = data[s.F].tocoo()
        g = data[s.G]
        lb = data[s.LOWER_BOUNDS]
        ub = data[s.UPPER_BOUNDS]

        results = {}
        try:
            kc = kn.KN_new()
        except Exception:  # Error in the Knitro.
            return {s.STATUS: s.SOLVER_ERROR}

        results[KNITRO.CONTEXT_KEY] = kc

        if not verbose:
            # Disable Knitro output.
            kn.KN_set_int_param(kc, kn.KN_PARAM_OUTLEV, kn.KN_OUTLEV_NONE)

        n_vars = 0
        if P is not None:
            n_vars = P.shape[0]
        elif q is not None:
            n_vars = q.shape[0]
        else:
            raise ValueError("No variables in the problem.")

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
        # - default: KN_VARTYPE_CONTINUOUS.
        # - binray: KN_VARTYPE_BINARY.
        # - integer: KN_VARTYPE_INTEGER.
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

        # Get the number of equality and inequality constraints.
        n_eqs, n_ineqs = A.shape[0], F.shape[0]
        n_cons = n_eqs + n_ineqs

        # Add the constraints to the problem.
        if n_cons > 0:
            kn.KN_add_cons(kc, n_cons)

        # Add linear equality and inequality constraints.
        if n_eqs > 0:
            cis = np.arange(n_eqs)
            kn.KN_set_con_eqbnds(kc, indexCons=cis, cEqBnds=b)
        if n_ineqs > 0:
            cis = n_eqs + np.arange(n_ineqs)
            kn.KN_set_con_upbnds(kc, indexCons=cis, cUpBnds=g)
        if n_eqs + n_ineqs > 0:
            D = sparse.vstack([A, F]).tocoo()
            cis, vis, coefs = D.row, D.col, D.data
            kn.KN_add_con_linear_struct(kc, indexCons=cis, indexVars=vis, coefs=coefs)

        # Set the initial values of the dual variables.
        if KNITRO.Y_INIT_KEY in solver_opts:
            i, vs = solver_opts[KNITRO.Y_INIT_KEY]
            kn.KN_set_con_dual_init_values(kc, indexCons=i, yInitVals=vs)

        # Set the objective function.
        # Set the linear part of the objective function.
        if q is not None:
            vis = np.arange(n_vars)
            kn.KN_add_obj_linear_struct(kc, indexVars=vis, coefs=q)

        # Set the quadratic part of the objective function.
        if P is not None and P.nnz != 0:
            Q = sparse.coo_matrix(0.5 * P)
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
        return CITATION_DICT["KNITRO"]
