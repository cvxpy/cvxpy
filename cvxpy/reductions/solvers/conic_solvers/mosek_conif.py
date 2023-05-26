"""
Copyright 2015 Enzo Busseti, 2017 Robin Verschueren, 2018 Riley Murray

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

import warnings
from collections import defaultdict

import numpy as np
import scipy as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor

__MSK_ENUM_PARAM_DEPRECATION__ = """
Using MOSEK constants to specify parameters is deprecated.
Use generic string names instead.
For example, replace mosek.iparam.num_threads with 'MSK_IPAR_NUM_THREADS'
"""


def vectorized_lower_tri_to_mat(v, dim):
    """
    :param v: a list of length (dim * (dim + 1) / 2)
    :param dim: the number of rows (equivalently, columns) in the output array.
    :return: Return the symmetric 2D array defined by taking "v" to
      specify its lower triangular entries.
    """
    rows, cols, vals = vectorized_lower_tri_to_triples(v, dim)
    A = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).toarray()
    d = np.diag(np.diag(A))
    A = A + A.T - d
    return A


def vectorized_lower_tri_to_triples(A: sp.sparse.coo_matrix | list[float] | np.ndarray, dim: int) \
        -> tuple[list[int], list[int], list[float]]:
    """
    Attributes
    ----------
    A : scipy.sparse.coo_matrix | list[float] | np.ndarray
        Contains the lower triangular entries of a symmetric matrix, flattened into a 1D array in
        column-major order.
    dim : int
        The number of rows (equivalently, columns) in the original matrix.

    Returns
    -------
    rows : list[int]
        The row indices of the entries in the original matrix.
    cols : list[int]
        The column indices of the entries in the original matrix.
    vals : list[float]
        The values of the entries in the original matrix.
    """

    if isinstance(A, sp.sparse.coo_matrix):
        vals = A.data
        flattened_cols = A.col
        # Ensure that the columns are sorted.
        if not np.all(flattened_cols[:-1] < flattened_cols[1:]):
            sort_idx = np.argsort(flattened_cols)
            vals = vals[sort_idx]
            flattened_cols = flattened_cols[sort_idx]
    elif isinstance(A, list):
        vals = A
        flattened_cols = np.arange(len(A))
    elif isinstance(A, np.ndarray):
        vals = list(A)
        flattened_cols = np.arange(len(A))
    else:
        raise TypeError(f"Expected A to be a coo_matrix, list, or ndarray, "
                        f"but got {type(A)} instead.")

    cum_cols = np.cumsum(np.arange(dim, 0, -1))
    rows, cols = [], []
    current_col = 0
    for v in flattened_cols:
        for c in range(current_col, dim):
            if v < cum_cols[c]:
                cols.append(c)
                prev_row = 0 if c == 0 else cum_cols[c - 1]
                rows.append(v - prev_row + c)
                break
            else:
                current_col += 1

    return rows, cols, vals


class MOSEK(ConicSolver):
    """ An interface for the Mosek solver.
    """

    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, PSD]
    EXP_CONE_ORDER = [2, 1, 0]
    DUAL_EXP_CONE_ORDER = [0, 1, 2]
    # Does not support MISDP.
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    """
    Note that MOSEK.SUPPORTED_CONSTRAINTS does not include the exponential cone
    by default. CVXPY will check for exponential cone support when
    "import_solver( ... )" or "accepts( ... )" is called.

    The cvxpy standard for the exponential cone is:
        K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
    Whenever a solver uses this convention, EXP_CONE_ORDER should be [0, 1, 2].

    MOSEK uses the convention:
        K_e = closure{(x,y,z) | x >= y * exp(z/y), x,y >= 0}.
    with this convention, EXP_CONE_ORDER should be should be [2, 1, 0].
    """

    def import_solver(self) -> None:
        """Imports the solver (updates the set of supported constraints, if applicable).
        """
        import mosek  # noqa F401

        if hasattr(mosek.conetype, 'pexp') and ExpCone not in MOSEK.SUPPORTED_CONSTRAINTS:
            MOSEK.SUPPORTED_CONSTRAINTS.append(ExpCone)
            MOSEK.SUPPORTED_CONSTRAINTS.append(PowCone3D)
            MOSEK.MI_SUPPORTED_CONSTRAINTS.append(ExpCone)
            MOSEK.MI_SUPPORTED_CONSTRAINTS.append(PowCone3D)

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def accepts(self, problem) -> bool:
        """Can the installed version of Mosek solve the problem?
        """
        # TODO check if is matrix stuffed.
        self.import_solver()
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in MOSEK.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    @staticmethod
    def psd_format_mat(constr):
        """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as MOSEK expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.

        This function differs from ``SCS.psd_format_mat`` only in that it does not
        apply sqrt(2) scaling on off-diagonal entries. This difference from SCS is
        necessary based on how we implement ``MOSEK.bar_data``.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1)//2

        row_arr = np.arange(0, entries)

        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = 1
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows*cols)
        scaled_lower_tri = sp.sparse.csc_matrix((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.sparse.csc_matrix((val_symm, (row_symm, col_symm)))

        return scaled_lower_tri @ symm_matrix

    @staticmethod
    def bar_data(A_psd, c_psd, K):
        # TODO: investigate how to transform or represent "A_psd" so that the following
        #  indexing and slicing operations are computationally cheap. Or just rewrite
        #  the function so that explicit slicing is not needed on a SciPy sparse matrix.
        n = A_psd.shape[0]
        c_bar_data, A_bar_data = [], []
        idx = 0
        for j, dim in enumerate(K[a2d.PSD]):  # psd variable index j.
            vec_len = dim * (dim + 1) // 2
            A_block = A_psd[:, idx:idx + vec_len]
            # ^ each row specifies a linear operator on PSD variable.
            for i in range(n):
                # A_row defines a symmetric matrix by where the first "order" entries
                #   gives the matrix's first column, the second "order-1" entries gives
                #   the matrix's second column (diagonal and below), and so on.
                A_row = A_block[i, :]
                if A_row.nnz == 0:
                    continue

                A_row_coo = A_row.tocoo()
                rows, cols, vals = vectorized_lower_tri_to_triples(A_row_coo, dim)
                A_bar_data.append((i, j, (rows, cols, vals)))

            c_block = c_psd[idx:idx + vec_len]
            rows, cols, vals = vectorized_lower_tri_to_triples(c_block, dim)
            c_bar_data.append((j, (rows, cols, vals)))
            idx += vec_len
        return A_bar_data, c_bar_data

    def apply(self, problem):
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        if problem.x.boolean_idx or problem.x.integer_idx:  # check if either list is empty
            data, inv_data = Slacks.apply(problem, [a2d.NONNEG])
        else:
            data, inv_data = Dualize.apply(problem)
            # need to do more to handle SDP.
            A, c, K = data[s.A], data[s.C], data['K_dir']
            num_psd = len(K[a2d.PSD])
            if num_psd > 0:
                idx = K[a2d.FREE] + K[a2d.NONNEG] + sum(K[a2d.SOC])
                total_psd = sum([d * (d+1) // 2 for d in K[a2d.PSD]])
                A_psd = A[:, idx:idx+total_psd]
                c_psd = c[idx:idx+total_psd]
                if (K[a2d.DUAL_EXP] == 0) and (K[a2d.DUAL_POW3D] == 0):
                    data[s.A] = A[:, :idx]
                    data[s.C] = c[:idx]
                else:
                    data[s.A] = sp.sparse.hstack([A[:, :idx], A[:, idx+total_psd:]])
                    data[s.C] = np.concatenate([c[:idx], c[idx+total_psd:]])
                A_bar_data, c_bar_data = MOSEK.bar_data(A_psd, c_psd, K)
                data['A_bar_data'] = A_bar_data
                data['c_bar_data'] = c_bar_data
            else:
                data['A_bar_data'] = []
                data['c_bar_data'] = []

        data[s.PARAM_PROB] = problem
        return data, inv_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import mosek

        if 'dualized' in data:
            if len(data[s.C]) == 0 and len(data['c_bar_data']) == 0:
                # primal problem was unconstrained minimization of a linear function.
                if np.linalg.norm(data[s.B]) > 0:
                    sol = Solution(s.INFEASIBLE, -np.inf, None, None, dict())
                    return {'sol': sol}
                else:
                    sol = Solution(s.OPTIMAL, 0.0, dict(), {s.EQ_DUAL: data[s.B]}, dict())
                    return {'sol': sol}
            else:
                env = mosek.Env()
                task = env.Task(0, 0)
                save_file = MOSEK.handle_options(env, task, verbose, solver_opts)
                task = MOSEK._build_dualized_task(task, data)
        else:
            if len(data[s.C]) == 0:
                sol = Solution(s.OPTIMAL, 0.0, dict(), dict(), dict())
                return {'sol': sol}
            else:
                env = mosek.Env()
                task = env.Task(0, 0)
                save_file = MOSEK.handle_options(env, task, verbose, solver_opts)
                task = MOSEK._build_slack_task(task, data)

        # Optimize the Mosek Task, and return the result.
        if save_file:
            task.writedata(save_file)
        task.optimize()

        if verbose:
            task.solutionsummary(mosek.streamtype.msg)

        return {'env': env, 'task': task, 'solver_options': solver_opts}

    @staticmethod
    def _build_dualized_task(task, data):
        """
        This function assumes "data" is formatted according to MOSEK.apply when the problem
        features no integer constraints. This dictionary should contain keys s.C, s.A, s.B,
        'K_dir', 'c_bar_data' and 'A_bar_data'.

        If the problem has no PSD constraints, then we construct a Task representing

           max{ c.T @ x : A @ x == b, x in K_dir }

        If the problem has PSD constraints, then the Task looks like

           max{ c.T @ x + c_bar(X_bars) : A @ x + A_bar(X_bars) == b, x in K_dir, X_bars PSD }

        In the above formulation, c_bar is effectively specified by a list of appropriately
        formatted symmetric matrices (one symmetric matrix for each PSD variable). A_bar
        is specified a collection of symmetric matrix data indexed by (i, j) where the j-th
        PSD variable contributes a certain scalar to the i-th linear equation in the system
        "A @ x + A_bar(X_bars) == b".
        """
        import mosek

        # problem data
        c, A, b, K = data[s.C], data[s.A], data[s.B], data['K_dir']
        n, m = A.shape
        task.appendvars(m)
        o = np.zeros(m)
        task.putvarboundlist(np.arange(m, dtype=int), [mosek.boundkey.fr] * m, o, o)
        task.appendcons(n)
        # objective
        task.putclist(np.arange(c.size, dtype=int), c)
        task.putobjsense(mosek.objsense.maximize)
        # equality constraints
        rows, cols, vals = sp.sparse.find(A)
        task.putaijlist(rows.tolist(), cols.tolist(), vals.tolist())
        task.putconboundlist(np.arange(n, dtype=int), [mosek.boundkey.fx] * n, b, b)
        # conic constraints
        idx = K[a2d.FREE]
        num_pos = K[a2d.NONNEG]
        if num_pos > 0:
            o = np.zeros(num_pos)
            task.putvarboundlist(np.arange(idx, idx + num_pos, dtype=int),
                                 [mosek.boundkey.lo] * num_pos, o, o)
            idx += num_pos
        num_soc = len(K[a2d.SOC])
        if num_soc > 0:
            cones = [mosek.conetype.quad] * num_soc
            task.appendconesseq(cones, [0] * num_soc, K[a2d.SOC], idx)
            idx += sum(K[a2d.SOC])
        num_dexp = K[a2d.DUAL_EXP]
        if num_dexp > 0:
            cones = [mosek.conetype.dexp] * num_dexp
            task.appendconesseq(cones, [0] * num_dexp, [3] * num_dexp, idx)
            idx += 3 * num_dexp
        num_dpow = len(K[a2d.DUAL_POW3D])
        if num_dpow > 0:
            cones = [mosek.conetype.dpow] * num_dpow
            task.appendconesseq(cones, K[a2d.DUAL_POW3D], [3] * num_dpow, idx)
            idx += 3 * num_dpow
        num_psd = len(K[a2d.PSD])
        if num_psd > 0:
            task.appendbarvars(K[a2d.PSD])
            psd_dims = np.array(K[a2d.PSD])
            for i, j, triples in data['A_bar_data']:
                order = psd_dims[j]
                operator_id = task.appendsparsesymmat(order, triples[0], triples[1], triples[2])
                task.putbaraij(i, j, [operator_id], [1.0])
            for j, triples in data['c_bar_data']:
                order = psd_dims[j]
                operator_id = task.appendsparsesymmat(order, triples[0], triples[1], triples[2])
                task.putbarcj(j, [operator_id], [1.0])
        return task

    @staticmethod
    def _build_slack_task(task, data):
        """
        This function assumes "data" is formatted by MOSEK.apply, and is only intended when
        the problem has integer constraints. As of MOSEK version 9.2, MOSEK does not support
        mixed-integer SDP. This implementation relies on that fact.

        "data" is a dict, keyed by s.C, s.A, s.B, 'K_dir', 'K_aff', s.BOOL_IDX and s.INT_IDX.
        The data 'K_aff' corresponds to constraints which MOSEK accepts as "A @ x <=_{K_aff}"
        (so-called "affine"  conic constraints), in contrast with constraints which must be stated
        as "x in K_dir" ("direct" conic constraints). As of MOSEK 9.2, the only allowed K_aff is
        the zero cone and the nonnegative orthant. All other constraints must be specified in a
        "direct" sense.

        The returned Task represents

            min{ c.T @ x : A @ x <=_{K_aff} b,  x in K_dir, x[bools] in {0,1}, x[ints] in Z }.
        """
        import mosek
        K_aff = data['K_aff']
        # K_aff keyed by a2d.ZERO, a2d.NONNEG
        c, A, b = data[s.C], data[s.A], data[s.B]
        # The rows of (A, b) go by a2d.ZERO and then a2d.NONNEG
        K_dir = data['K_dir']
        # Components of the vector "x" are constrained in the order
        # a2d.FREE, then a2d.SOC, then a2d.EXP. PSD is not supported.
        m, n = A.shape
        task.appendvars(n)
        o = np.zeros(n)
        task.putvarboundlist(np.arange(n, dtype=int), [mosek.boundkey.fr] * n, o, o)
        task.appendcons(m)
        # objective
        task.putclist(np.arange(n, dtype=int), c)
        task.putobjsense(mosek.objsense.minimize)
        # elementwise constraints
        rows, cols, vals = sp.sparse.find(A)
        task.putaijlist(rows, cols, vals)
        eq_keys = [mosek.boundkey.fx] * K_aff[a2d.ZERO]
        ineq_keys = [mosek.boundkey.up] * K_aff[a2d.NONNEG]
        task.putconboundlist(np.arange(m, dtype=int), eq_keys + ineq_keys, b, b)
        # conic constraints
        idx = K_dir[a2d.FREE]
        num_soc = len(K_dir[a2d.SOC])
        if num_soc > 0:
            conetypes = [mosek.conetype.quad] * num_soc
            task.appendconesseq(conetypes, [0] * num_soc, K_dir[a2d.SOC], idx)
            idx += sum(K_dir[a2d.SOC])
        num_exp = K_dir[a2d.EXP]
        if num_exp > 0:
            conetypes = [mosek.conetype.pexp] * num_exp
            task.appendconesseq(conetypes, [0] * num_exp, [3] * num_exp, idx)
            idx += 3*num_exp
        num_pow = len(K_dir[a2d.POW3D])
        if num_pow > 0:
            conetypes = [mosek.conetype.ppow] * num_pow
            task.appendconesseq(conetypes, K_dir[a2d.POW3D], [3] * num_pow, idx)
            idx += 3*num_pow
        # integrality constraints
        num_bool = len(data[s.BOOL_IDX])
        num_int = len(data[s.INT_IDX])
        vartypes = [mosek.variabletype.type_int] * (num_bool + num_int)
        task.putvartypelist(data[s.INT_IDX] + data[s.BOOL_IDX], vartypes)
        if num_bool > 0:
            task.putvarboundlist(data[s.BOOL_IDX], [mosek.boundkey.ra] * num_bool,
                                 [0] * num_bool, [1] * num_bool)
        return task

    def invert(self, solver_output, inverse_data):
        """
        This function parses data from the MOSEK Task as though we only cared about the
        Task *exactly* as stated (i.e. we are indifferent to whether the Task represents
        the dual formulation to an earlier CVXPY problem). Once we have parsed that data
        into a Solution object called "raw_sol", we call the appropriate invert-step of
        the dualize or slack reduction to obtain a final result in terms of CVXPY's
        standard-form cone programs.
        """
        if 'sol' in solver_output:
            # The presence of this key means the original problem was somehow degenerate
            # (e.g. no variables, or no constraints). The MOSEK.solve_via_data function
            # automatically constructions appropriate solutions for these situations. So
            # in this case we only need to invert the transformations from MOSEK.apply.
            sol = solver_output['sol']
            if 'dualized' in inverse_data:
                sol = Dualize.invert(sol, inverse_data)
            else:
                sol = Slacks.invert(sol, inverse_data)
            return sol
        # If we reach this point in the code, then we actually called MOSEK's optimizer,
        # and we need to properly parse the result.

        import mosek

        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.integer_optimal: s.OPTIMAL,
                      mosek.solsta.prim_feas: s.OPTIMAL_INACCURATE,    # for integer problems
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED}
        # "Near" statuses only up to Mosek 8.1
        if hasattr(mosek.solsta, 'near_optimal'):
            STATUS_MAP[mosek.solsta.near_optimal] = s.OPTIMAL_INACCURATE
            STATUS_MAP[mosek.solsta.near_integer_optimal] = s.OPTIMAL_INACCURATE
            STATUS_MAP[mosek.solsta.near_prim_infeas_cer] = s.INFEASIBLE_INACCURATE
            STATUS_MAP[mosek.solsta.near_dual_infeas_cer] = s.UNBOUNDED_INACCURATE
        STATUS_MAP = defaultdict(lambda: s.SOLVER_ERROR, STATUS_MAP)

        env = solver_output['env']
        task = solver_output['task']
        solver_opts = solver_output['solver_options']
        simplex_algs = [
            mosek.optimizertype.primal_simplex,
            mosek.optimizertype.dual_simplex,
        ]
        current_optimizer = task.getintparam(mosek.iparam.optimizer)
        bfs_active = "bfs" in solver_opts and solver_opts["bfs"] and task.getnumcone() == 0

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif current_optimizer in simplex_algs or bfs_active:
            sol_type = mosek.soltype.bas  # the basic feasible solution
        else:
            sol_type = mosek.soltype.itr  # the solution found via interior point method

        prim_vars = None
        dual_vars = None
        problem_status = task.getprosta(sol_type)
        if sol_type == mosek.soltype.itg and problem_status == mosek.prosta.prim_infeas:
            status = s.INFEASIBLE
            prob_val = np.inf
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]
            prob_val = np.NaN
            if status in s.SOLUTION_PRESENT:
                prob_val = task.getprimalobj(sol_type)
                K = inverse_data['K_dir']
                prim_vars = MOSEK.recover_primal_variables(task, sol_type, K)
                dual_vars = MOSEK.recover_dual_variables(task, sol_type)
        attr = {s.SOLVE_TIME: task.getdouinf(mosek.dinfitem.optimizer_time)}
        raw_sol = Solution(status, prob_val, prim_vars, dual_vars, attr)

        if task.getobjsense() == mosek.objsense.maximize:
            sol = Dualize.invert(raw_sol, inverse_data)
        else:
            sol = Slacks.invert(raw_sol, inverse_data)

        # Delete the mosek Task and Environment
        task.__exit__(None, None, None)
        env.__exit__(None, None, None)

        return sol

    @staticmethod
    def recover_dual_variables(task, sol):
        # This function is only designed to recover dual variables
        # when the "dualize" transformation has been applied.
        # A problem is dualized if and only if it has no discrete variables.
        if task.getnumintvar() == 0:
            dual_vars = dict()
            dual_var = [0.] * task.getnumcon()
            task.gety(sol, dual_var)
            dual_vars[s.EQ_DUAL] = np.array(dual_var)
            # We only need to recover dual variables related to the equality constraints.
            # Dual variables related to the "direct" conic constraints are not needed when
            # inverting the solution to undo dualization.
        else:
            dual_vars = None
        return dual_vars

    @staticmethod
    def recover_primal_variables(task, sol, K_dir):
        # This function applies both when slacks are introduced, and
        # when the problem is dualized.
        prim_vars = dict()
        idx = 0
        m_free = K_dir[a2d.FREE]
        if m_free > 0:
            temp = [0.] * m_free
            task.getxxslice(sol, idx, len(temp), temp)
            prim_vars[a2d.FREE] = np.array(temp)
            idx += m_free
        if task.getnumintvar() > 0:
            return prim_vars  # Skip the slack variables.
        m_pos = K_dir[a2d.NONNEG]
        if m_pos > 0:
            temp = [0.] * m_pos
            task.getxxslice(sol, idx, idx + m_pos, temp)
            prim_vars[a2d.NONNEG] = np.array(temp)
            idx += m_pos
        num_soc = len(K_dir[a2d.SOC])
        if num_soc > 0:
            soc_vars = []
            for dim in K_dir[a2d.SOC]:
                temp = [0.] * dim
                task.getxxslice(sol, idx, idx + dim, temp)
                soc_vars.append(np.array(temp))
                idx += dim
            prim_vars[a2d.SOC] = soc_vars
        num_dexp = K_dir[a2d.DUAL_EXP]
        if num_dexp > 0:
            temp = [0.] * (3 * num_dexp)
            task.getxxslice(sol, idx, idx + len(temp), temp)
            temp = np.array(temp)
            perm = expcone_permutor(num_dexp, MOSEK.EXP_CONE_ORDER)
            prim_vars[a2d.DUAL_EXP] = temp[perm]
            idx += (3 * num_dexp)
        num_dpow = len(K_dir[a2d.DUAL_POW3D])
        if num_dpow > 0:
            temp = [0.] * (3 * num_dpow)
            task.getxxslice(sol, idx, idx + len(temp), temp)
            temp = np.array(temp)
            prim_vars[a2d.DUAL_POW3D] = temp
            idx += (3 * num_dpow)
        num_psd = len(K_dir[a2d.PSD])
        if num_psd > 0:
            psd_vars = []
            for j, dim in enumerate(K_dir[a2d.PSD]):
                xj = [0.] * (dim * (dim + 1) // 2)
                task.getbarxj(sol, j, xj)
                psd_vars.append(vectorized_lower_tri_to_mat(xj, dim))
            prim_vars[a2d.PSD] = psd_vars
        return prim_vars

    @staticmethod
    def handle_options(env, task, verbose: bool, solver_opts):
        # If verbose, then set default logging parameters.
        import mosek

        if verbose:

            def streamprinter(text):
                s.LOGGER.info(text.rstrip('\n'))

            print('\n')
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)

        # Parse all user-specified parameters (override default logging
        # parameters if applicable).
        kwargs = sorted(solver_opts.keys())
        save_file = None
        bfs = False
        if 'mosek_params' in kwargs:
            # Issue a warning if Mosek enums are used as parameter names / keys
            if any(isinstance(param, mosek.iparam) or
                   isinstance(param, mosek.dparam) or
                   isinstance(param, mosek.sparam) for param in solver_opts['mosek_params'].keys()):
                warnings.warn(__MSK_ENUM_PARAM_DEPRECATION__, DeprecationWarning)
                warnings.warn(__MSK_ENUM_PARAM_DEPRECATION__, UserWarning)
            # Now set parameters
            for param, value in solver_opts['mosek_params'].items():
                if isinstance(param, str):
                    # Parameters are set through generic string names (recommended)
                    param = param.strip()
                    if isinstance(value, str):
                        # The value is also a string.
                        # Examples: "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL"
                        #           "MSK_DPAR_CO_TOL_PFEAS": "1.0e-9"
                        task.putparam(param, value)
                    else:
                        # Otherwise we assume the value is of correct type
                        if param.startswith("MSK_DPAR_"):
                            task.putnadouparam(param, value)
                        elif param.startswith("MSK_IPAR_"):
                            task.putnaintparam(param, value)
                        elif param.startswith("MSK_SPAR_"):
                            task.putnastrparam(param, value)
                        else:
                            raise ValueError("Invalid MOSEK parameter '%s'." % param)
                else:
                    # Parameters are set through Mosek enums (deprecated)
                    if isinstance(param, mosek.dparam):
                        task.putdouparam(param, value)
                    elif isinstance(param, mosek.iparam):
                        task.putintparam(param, value)
                    elif isinstance(param, mosek.sparam):
                        task.putstrparam(param, value)
                    else:
                        raise ValueError("Invalid MOSEK parameter '%s'." % param)
            kwargs.remove('mosek_params')
        if 'save_file' in kwargs:
            save_file = solver_opts['save_file']
            kwargs.remove('save_file')
        if 'bfs' in kwargs:
            bfs = solver_opts['bfs']
            kwargs.remove('bfs')
        if kwargs:
            raise ValueError("Invalid keyword-argument '%s'" % kwargs[0])

        # Decide whether basis identification is needed for intpnt solver
        # This is only required if solve() was called with bfs=True
        if bfs:
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.always)
        else:
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        return save_file
