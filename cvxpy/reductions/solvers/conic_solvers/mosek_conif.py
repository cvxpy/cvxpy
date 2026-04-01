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

from collections import defaultdict

import numpy as np
import scipy as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D, PowConeND, SvecPSD
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import (
    DUAL_EXP,
    DUAL_POW3D,
)
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.psd_utils import TriangleKind
from cvxpy.utilities.warn import CvxpyDeprecationWarning, warn

__MSK_ENUM_PARAM_DEPRECATION__ = """
Using MOSEK constants to specify parameters is deprecated.
Use generic string names instead.
For example, replace mosek.iparam.num_threads with 'MSK_IPAR_NUM_THREADS'
"""


def vectorized_lower_tri_to_triples(
    A: sp.sparse.coo_matrix | sp.sparse.sparray | list[float] | np.ndarray,
    dim: int,
) -> tuple[list[int], list[int], list[float]]:
    """
    Attributes
    ----------
    A : scipy.sparse.coo_matrix | list[float] | np.ndarray
        Contains the lower triangular entries of a symmetric matrix,
        flattened into a 1D array in column-major order.
    dim : int
        The number of rows (equivalently, columns) in the original
        matrix.

    Returns
    -------
    rows : list[int]
        The row indices of the entries in the original matrix.
    cols : list[int]
        The column indices of the entries in the original matrix.
    vals : list[float]
        The values of the entries in the original matrix.
    """
    if sp.sparse.issparse(A):
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
        raise TypeError(
            f"Expected A to be a coo_matrix, list, or ndarray, "
            f"but got {type(A)} instead."
        )

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
    """An interface for the Mosek solver."""

    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC, SvecPSD, ExpCone, PowCone3D, PowConeND,
    ]
    PSD_TRIANGLE_KIND = TriangleKind.LOWER
    PSD_SQRT2_SCALING = False
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC, ExpCone, PowCone3D, PowConeND,
    ]

    # MOSEK uses the convention K_e = closure{(x,y,z) | x >= y*exp(z/y)},
    # so EXP_CONE_ORDER is [2, 1, 0] (reversed from the CVXPY standard).
    EXP_CONE_ORDER = [2, 1, 0]
    DUAL_EXP_CONE_ORDER = [0, 1, 2]

    def import_solver(self) -> None:
        """Imports the solver."""
        import mosek  # noqa F401

    def name(self):
        """The name of the solver."""
        return s.MOSEK

    def should_dualize(self, problem_form) -> bool:
        """Dualize when PSD constraints are present and problem is
        continuous."""
        if problem_form.is_mixed_integer():
            return False
        cones = problem_form.cones(quad_obj=False)
        return PSD in cones

    def accepts(self, problem) -> bool:
        """Can the installed version of Mosek solve the problem?"""
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

    # ------------------------------------------------------------------
    # bar_data — used only by the dualized (PSD) path
    # ------------------------------------------------------------------

    @staticmethod
    def bar_data(A_psd, c_psd, K):
        # TODO: investigate how to transform or represent "A_psd" so
        #  that the following indexing and slicing operations are
        #  computationally cheap. Or just rewrite the function so that
        #  explicit slicing is not needed on a SciPy sparse matrix.
        n = A_psd.shape[0]
        c_bar_data, A_bar_data = [], []
        idx = 0
        for j, dim in enumerate(K[a2d.PSD]):  # psd variable index j.
            vec_len = dim * (dim + 1) // 2
            A_block = A_psd[:, idx:idx + vec_len]
            # ^ each row specifies a linear operator on PSD variable.
            for i in range(n):
                # A_row defines a symmetric matrix where the first
                # "order" entries give the matrix's first column, the
                # second "order-1" entries give the second column
                # (diagonal and below), and so on.
                A_row = A_block[[i], :]
                if A_row.nnz == 0:
                    continue
                A_row_coo = A_row.tocoo()
                rows, cols, vals = vectorized_lower_tri_to_triples(
                    A_row_coo, dim,
                )
                A_bar_data.append((i, j, (rows, cols, vals)))
            c_block = c_psd[idx:idx + vec_len]
            rows, cols, vals = vectorized_lower_tri_to_triples(
                c_block, dim,
            )
            c_bar_data.append((j, (rows, cols, vals)))
            idx += vec_len
        return A_bar_data, c_bar_data

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(self, problem):
        if problem.dualized:
            return self._apply_dualized(problem)
        return self._apply_acc(problem)

    def _apply_dualized(self, problem):
        """Dualized path — used when PSD constraints are present.

        Extracts A, b, c, d from the ParamConeProg and transposes into
        the dual form that ``_build_dualized_task`` expects.  PSD
        columns are separated into bar_data for MOSEK's bar variable API.
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(
            problem,
        )
        c, d, A, b = problem.apply_parameters()
        Kp = problem.cone_dims
        Kd = {
            a2d.FREE: Kp.zero,
            a2d.NONNEG: Kp.nonneg,
            a2d.SOC: Kp.soc,
            a2d.PSD: Kp.psd,
            a2d.DUAL_EXP: Kp.exp,
            a2d.DUAL_POW3D: Kp.p3d,
        }
        data[s.A] = A.T
        data[s.B] = c
        data[s.C] = -b
        data['K_dir'] = Kd
        data['dualized'] = True
        inv_data[s.OFFSET] = d
        # Extract PSD columns into bar_data for MOSEK's bar variable API.
        num_psd = len(Kd[a2d.PSD])
        if num_psd > 0:
            At, ct = data[s.A], data[s.C]
            idx = (
                Kd[a2d.FREE]
                + Kd[a2d.NONNEG]
                + sum(Kd[a2d.SOC])
            )
            total_psd = sum(
                dim * (dim + 1) // 2 for dim in Kd[a2d.PSD]
            )
            A_psd = At[:, idx:idx + total_psd]
            c_psd = ct[idx:idx + total_psd]
            if Kd[a2d.DUAL_EXP] == 0 and not Kd[a2d.DUAL_POW3D]:
                data[s.A] = At[:, :idx]
                data[s.C] = ct[:idx]
            else:
                data[s.A] = sp.sparse.hstack(
                    [At[:, :idx], At[:, idx + total_psd:]],
                )
                data[s.C] = np.concatenate(
                    [ct[:idx], ct[idx + total_psd:]],
                )
            A_bar_data, c_bar_data = MOSEK.bar_data(
                A_psd, c_psd, Kd,
            )
            data['A_bar_data'] = A_bar_data
            data['c_bar_data'] = c_bar_data
        else:
            data['A_bar_data'] = []
            data['c_bar_data'] = []
        return data, inv_data

    def _apply_acc(self, problem):
        """ACC path — primal form, cones on constraints."""
        problem, data, inv_data = self._prepare_data_and_inv_data(
            problem,
        )
        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        data[s.A] = A
        data[s.B] = b
        inv_data[s.OFFSET] = d
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds
        data[s.BOOL_IDX] = [
            int(t[0]) for t in problem.x.boolean_idx
        ]
        data[s.INT_IDX] = [
            int(t[0]) for t in problem.x.integer_idx
        ]
        return data, inv_data

    # ------------------------------------------------------------------
    # solve_via_data
    # ------------------------------------------------------------------

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool,
        solver_opts, solver_cache=None,
    ):
        import mosek

        if 'dualized' in data:
            # Dualized (PSD) path
            if (
                len(data[s.C]) == 0
                and len(data['c_bar_data']) == 0
            ):
                if np.linalg.norm(data[s.B]) > 0:
                    sol = Solution(
                        s.INFEASIBLE, -np.inf, None, None, dict(),
                    )
                    return {'sol': sol}
                else:
                    sol = Solution(
                        s.OPTIMAL, 0.0, dict(),
                        {s.EQ_DUAL: data[s.B]}, dict(),
                    )
                    return {'sol': sol}
            task = mosek.Task()
            solver_opts = MOSEK.handle_options(
                task, verbose, solver_opts,
            )
            task = MOSEK._build_dualized_task(task, data)
        else:
            # ACC (primal) path
            if len(data[s.C]) == 0:
                sol = Solution(
                    s.OPTIMAL, 0.0, dict(), dict(), dict(),
                )
                return {'sol': sol}
            task = mosek.Task()
            solver_opts = MOSEK.handle_options(
                task, verbose, solver_opts,
            )
            task = MOSEK._build_task(task, data)

        # Save the task to a file if requested.
        save_file = solver_opts['save_file']
        if save_file:
            task.writedata(save_file)

        # Optimize the Mosek Task, and return the result.
        rescode = task.optimize()

        if rescode == mosek.rescode.trm_max_time:
            warn(
                "Optimization terminated by time limit; "
                "solution may be imprecise or absent.",
            )

        if verbose:
            task.solutionsummary(mosek.streamtype.msg)

        return {'task': task, 'solver_options': solver_opts}

    # ------------------------------------------------------------------
    # _build_dualized_task — existing dualized (PSD) path
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dualized_task(task, data):
        """Build a MOSEK task for the dualized formulation.

        This is used when the problem has PSD constraints and no
        integer variables.  The task represents

            max{ c'x + c_bar(X_bars) :
                 A x + A_bar(X_bars) = b,
                 x in K_dir, X_bars PSD }
        """
        import mosek

        c, A, b, K = (
            data[s.C], data[s.A], data[s.B], data['K_dir'],
        )
        n, m = A.shape
        task.appendvars(m)
        o = np.zeros(m)
        task.putvarboundlist(
            np.arange(m, dtype=np.int32),
            [mosek.boundkey.fr] * m, o, o,
        )
        task.appendcons(n)
        # objective
        task.putclist(np.arange(c.size, dtype=np.int32), c)
        task.putobjsense(mosek.objsense.maximize)
        # equality constraints
        rows, cols, vals = sp.sparse.find(A)
        task.putaijlist(
            rows.tolist(), cols.tolist(), vals.tolist(),
        )
        task.putconboundlist(
            np.arange(n, dtype=np.int32),
            [mosek.boundkey.fx] * n, b, b,
        )
        # conic constraints
        idx = K[a2d.FREE]
        num_pos = K[a2d.NONNEG]
        if num_pos > 0:
            o = np.zeros(num_pos)
            task.putvarboundlist(
                np.arange(idx, idx + num_pos, dtype=np.int32),
                [mosek.boundkey.lo] * num_pos, o, o,
            )
            idx += num_pos
        num_soc = len(K[a2d.SOC])
        if num_soc > 0:
            cones = [mosek.conetype.quad] * num_soc
            task.appendconesseq(
                cones, [0] * num_soc, K[a2d.SOC], idx,
            )
            idx += sum(K[a2d.SOC])
        num_rsoc = len(K[a2d.RSOC])
        if num_rsoc > 0:
            cones = [mosek.conetype.rquad] * num_rsoc
            task.appendconesseq(
                cones, [0] * num_rsoc, K[a2d.RSOC], idx,
            )
            idx += sum(K[a2d.RSOC])
        num_dexp = K[a2d.DUAL_EXP]
        if num_dexp > 0:
            cones = [mosek.conetype.dexp] * num_dexp
            task.appendconesseq(
                cones, [0] * num_dexp, [3] * num_dexp, idx,
            )
            idx += 3 * num_dexp
        num_dpow = len(K[a2d.DUAL_POW3D])
        if num_dpow > 0:
            cones = [mosek.conetype.dpow] * num_dpow
            task.appendconesseq(
                cones, K[a2d.DUAL_POW3D], [3] * num_dpow, idx,
            )
            idx += 3 * num_dpow
        num_psd = len(K[a2d.PSD])
        if num_psd > 0:
            task.appendbarvars(K[a2d.PSD])
            psd_dims = np.array(K[a2d.PSD])
            for i, j, triples in data['A_bar_data']:
                order = psd_dims[j]
                operator_id = task.appendsparsesymmat(
                    order, triples[0], triples[1], triples[2],
                )
                task.putbaraij(i, j, [operator_id], [1.0])
            for j, triples in data['c_bar_data']:
                order = psd_dims[j]
                operator_id = task.appendsparsesymmat(
                    order, triples[0], triples[1], triples[2],
                )
                task.putbarcj(j, [operator_id], [1.0])
        return task

    # ------------------------------------------------------------------
    # ACC helpers — primal-form path
    # ------------------------------------------------------------------

    @staticmethod
    def _add_afe_acc(task, A, b, row, dim, domain_idx):
        """Add an ACC for rows A[row:row+dim, :] x + b[row:row+dim]
        in the given domain."""
        afe_base = task.getnumafe()
        task.appendafes(dim)
        A_block = A[row:row + dim, :]
        if sp.sparse.issparse(A_block):
            A_coo = A_block.tocoo()
            rows_i = A_coo.row
            cols_j = A_coo.col
            vals = A_coo.data
        else:
            rows_i, cols_j = np.nonzero(A_block)
            vals = np.asarray(
                A_block[rows_i, cols_j],
            ).ravel()
        if len(vals) > 0:
            task.putafefentrylist(
                (rows_i + afe_base).tolist(),
                cols_j.tolist(),
                vals.tolist(),
            )
        task.putafegslice(
            afe_base, afe_base + dim,
            b[row:row + dim].tolist(),
        )
        task.appendacc(
            domain_idx,
            list(range(afe_base, afe_base + dim)),
            None,
        )

    @staticmethod
    def _build_task(task, data):
        """Build a MOSEK task using the ACC (AFE) API.

        This is the primal-form path for problems without PSD
        constraints.  Conic constraints are expressed via Affine
        Conic Constraints (ACC).
        """
        import mosek

        c = data[s.C]
        A = data[s.A]
        b = data[s.B]
        cone_dims = data[ConicSolver.DIMS]
        n = c.size

        # Variables
        task.appendvars(n)

        # Variable bounds
        if (
            data[s.LOWER_BOUNDS] is not None
            and data[s.UPPER_BOUNDS] is not None
        ):
            bl = data[s.LOWER_BOUNDS].copy()
            bu = data[s.UPPER_BOUNDS].copy()
            bk = np.empty(n, dtype=object)
            mask = np.isfinite([bl, bu])
            bk[(~mask[0]) & (~mask[1])] = mosek.boundkey.fr
            bk[(~mask[0]) & mask[1]] = mosek.boundkey.up
            bk[mask[0] & (~mask[1])] = mosek.boundkey.lo
            bk[mask[0] & mask[1]] = mosek.boundkey.ra
            bl[~mask[0]] = 0.0
            bu[~mask[1]] = 0.0
            task.putvarboundlist(
                np.arange(n, dtype=np.int32), list(bk), bl, bu,
            )
        else:
            o = np.zeros(n)
            task.putvarboundlist(
                np.arange(n, dtype=np.int32),
                [mosek.boundkey.fr] * n, o, o,
            )

        # Objective: min c'x
        task.putclist(np.arange(n, dtype=np.int32), c)
        task.putobjsense(mosek.objsense.minimize)

        row = 0

        # Zero cone -> MOSEK linear equality constraints
        num_eq = cone_dims.zero
        if num_eq > 0:
            con_offset = task.getnumcon()
            task.appendcons(num_eq)
            A_eq = A[row:row + num_eq, :]
            if sp.sparse.issparse(A_eq):
                A_coo = A_eq.tocoo()
                eq_rows = A_coo.row
                eq_cols = A_coo.col
                eq_vals = A_coo.data
            else:
                eq_rows, eq_cols = np.nonzero(A_eq)
                eq_vals = np.asarray(
                    A_eq[eq_rows, eq_cols],
                ).ravel()
            if len(eq_vals) > 0:
                task.putaijlist(
                    (eq_rows + con_offset).tolist(),
                    eq_cols.tolist(),
                    eq_vals.tolist(),
                )
            bounds = (-b[row:row + num_eq]).tolist()
            task.putconboundlist(
                np.arange(
                    con_offset, con_offset + num_eq,
                    dtype=np.int32,
                ),
                [mosek.boundkey.fx] * num_eq,
                bounds, bounds,
            )
            row += num_eq

        # NonNeg cone -> MOSEK linear inequality constraints
        num_nn = cone_dims.nonneg
        if num_nn > 0:
            con_offset = task.getnumcon()
            task.appendcons(num_nn)
            A_nn = A[row:row + num_nn, :]
            if sp.sparse.issparse(A_nn):
                A_coo = A_nn.tocoo()
                nn_rows = A_coo.row
                nn_cols = A_coo.col
                nn_vals = A_coo.data
            else:
                nn_rows, nn_cols = np.nonzero(A_nn)
                nn_vals = np.asarray(
                    A_nn[nn_rows, nn_cols],
                ).ravel()
            if len(nn_vals) > 0:
                task.putaijlist(
                    (nn_rows + con_offset).tolist(),
                    nn_cols.tolist(),
                    nn_vals.tolist(),
                )
            lb = (-b[row:row + num_nn]).tolist()
            ub = [0.0] * num_nn
            task.putconboundlist(
                np.arange(
                    con_offset, con_offset + num_nn,
                    dtype=np.int32,
                ),
                [mosek.boundkey.lo] * num_nn,
                lb, ub,
            )
            row += num_nn

        # SOC cones via ACC
        for dim in cone_dims.soc:
            MOSEK._add_afe_acc(
                task, A, b, row, dim,
                task.appendquadraticconedomain(dim),
            )
            row += dim

        # RSOC cones via ACC
        for dim in cone_dims.rsoc:
            MOSEK._add_afe_acc(
                task, A, b, row, dim,
                task.appendrquadraticconedomain(dim),
            )
            row += dim

        # PSD cones via ACC (svec format with sqrt(2) scaling)
        for psd_dim in cone_dims.psd:
            vec_len = psd_dim * (psd_dim + 1) // 2
            MOSEK._add_afe_acc(
                task, A, b, row, vec_len,
                task.appendsvecpsdconedomain(vec_len),
            )
            row += vec_len

        # Exp cones via ACC
        for _ in range(cone_dims.exp):
            MOSEK._add_afe_acc(
                task, A, b, row, 3,
                task.appendprimalexpconedomain(),
            )
            row += 3

        # Pow3D cones via ACC
        for alpha in cone_dims.p3d:
            dom = task.appendprimalpowerconedomain(
                3, [alpha, 1.0 - alpha],
            )
            MOSEK._add_afe_acc(task, A, b, row, 3, dom)
            row += 3

        # PowND cones via ACC
        for alpha in cone_dims.pnd:
            dim = len(alpha) + 1
            dom = task.appendprimalpowerconedomain(
                dim, list(alpha),
            )
            MOSEK._add_afe_acc(task, A, b, row, dim, dom)
            row += dim

        # Integer constraints
        bool_idx = data[s.BOOL_IDX]
        int_idx = data[s.INT_IDX]
        num_bool = len(bool_idx)
        num_int = len(int_idx)
        if num_bool + num_int > 0:
            vartypes = [mosek.variabletype.type_int] * (
                num_bool + num_int
            )
            task.putvartypelist(int_idx + bool_idx, vartypes)
        if num_bool > 0:
            task.putvarboundlist(
                bool_idx, [mosek.boundkey.ra] * num_bool,
                [0] * num_bool, [1] * num_bool,
            )

        return task

    # ------------------------------------------------------------------
    # invert
    # ------------------------------------------------------------------

    def invert(self, solver_output, inverse_data):
        """Parse MOSEK solution and invert reductions.

        Dispatches to the dualized invert path (when the task was a
        maximization built by _build_dualized_task) or the ACC invert
        path (when the task was a minimization built by _build_task).
        """
        if 'sol' in solver_output:
            # Degenerate problem handled in solve_via_data.
            return solver_output['sol']

        import mosek

        task = solver_output['task']

        if task.getobjsense() == mosek.objsense.maximize:
            return self._invert_dualized(
                solver_output, inverse_data,
            )
        return self._invert_acc(solver_output, inverse_data)

    def _invert_dualized(self, solver_output, inverse_data):
        """Invert path for the dualized (PSD) formulation."""
        import mosek

        STATUS_MAP = {
            mosek.solsta.optimal: s.OPTIMAL,
            mosek.solsta.integer_optimal: s.OPTIMAL,
            mosek.solsta.prim_feas: s.OPTIMAL_INACCURATE,
            mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
            mosek.solsta.dual_infeas_cer: s.UNBOUNDED,
        }

        task = solver_output['task']
        solver_opts = solver_output['solver_options']

        if solver_opts['accept_unknown']:
            STATUS_MAP[mosek.solsta.unknown] = s.OPTIMAL_INACCURATE

        STATUS_MAP = defaultdict(
            lambda: s.SOLVER_ERROR, STATUS_MAP,
        )

        simplex_algs = [
            mosek.optimizertype.primal_simplex,
            mosek.optimizertype.dual_simplex,
        ]
        current_optimizer = task.getintparam(
            mosek.iparam.optimizer,
        )
        bfs_active = (
            "bfs" in solver_opts
            and solver_opts["bfs"]
            and task.getnumcone() == 0
        )

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif current_optimizer in simplex_algs or bfs_active:
            sol_type = mosek.soltype.bas
        else:
            sol_type = mosek.soltype.itr

        prim_vars = None
        dual_vars = None
        problem_status = task.getprosta(sol_type)
        attr = {
            s.SOLVE_TIME: task.getdouinf(
                mosek.dinfitem.optimizer_time,
            ),
            s.NUM_ITERS: (
                task.getintinf(mosek.iinfitem.intpnt_iter)
                + task.getintinf(mosek.iinfitem.sim_primal_iter)
                + task.getintinf(mosek.iinfitem.sim_dual_iter)
                + task.getintinf(mosek.iinfitem.mio_num_relax)
            ),
            s.EXTRA_STATS: {
                "mio_intpnt_iter": task.getlintinf(
                    mosek.liinfitem.mio_intpnt_iter,
                ),
                "mio_simplex_iter": task.getlintinf(
                    mosek.liinfitem.mio_simplex_iter,
                ),
            },
        }

        if (
            sol_type == mosek.soltype.itg
            and problem_status == mosek.prosta.prim_infeas
        ):
            status = s.INFEASIBLE
            prob_val = np.inf
        elif problem_status == mosek.prosta.dual_infeas:
            status = s.UNBOUNDED
            prob_val = -np.inf
            K = inverse_data["K_dir"]
            prim_vars = MOSEK.recover_primal_variables(
                task, sol_type, K,
            )
            dual_vars = MOSEK.recover_dual_variables(
                task, sol_type,
            )
            raw_iis_sol = Solution(
                s.OPTIMAL, prob_val, prim_vars, dual_vars, attr,
            )
            attr[s.EXTRA_STATS] = {
                "IIS": raw_iis_sol.dual_vars,
            }
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]
            prob_val = np.nan
            if status in s.SOLUTION_PRESENT:
                prob_val = task.getprimalobj(sol_type)
                K = inverse_data["K_dir"]
                prim_vars = MOSEK.recover_primal_variables(
                    task, sol_type, K,
                )
                dual_vars = MOSEK.recover_dual_variables(
                    task, sol_type,
                )

        sol = Solution(
            status, prob_val, prim_vars, dual_vars, attr,
        )
        task.__exit__(None, None, None)
        return sol

    def _invert_acc(self, solver_output, inverse_data):
        """Invert path for the ACC (primal-form) formulation."""
        import mosek

        STATUS_MAP = {
            mosek.solsta.optimal: s.OPTIMAL,
            mosek.solsta.integer_optimal: s.OPTIMAL,
            mosek.solsta.prim_feas: s.OPTIMAL_INACCURATE,
            mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
            mosek.solsta.dual_infeas_cer: s.UNBOUNDED,
        }

        task = solver_output['task']
        solver_opts = solver_output['solver_options']

        if solver_opts['accept_unknown']:
            STATUS_MAP[mosek.solsta.unknown] = s.OPTIMAL_INACCURATE

        STATUS_MAP = defaultdict(
            lambda: s.SOLVER_ERROR, STATUS_MAP,
        )

        cone_dims = inverse_data[self.DIMS]

        # Determine solution type
        simplex_algs = [
            mosek.optimizertype.primal_simplex,
            mosek.optimizertype.dual_simplex,
        ]
        current_optimizer = task.getintparam(
            mosek.iparam.optimizer,
        )
        bfs_active = (
            solver_opts.get("bfs", False)
            and task.getnumacc() == 0
        )

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif current_optimizer in simplex_algs or bfs_active:
            sol_type = mosek.soltype.bas
        else:
            sol_type = mosek.soltype.itr

        # Solver attributes
        problem_status = task.getprosta(sol_type)
        attr = {
            s.SOLVE_TIME: task.getdouinf(
                mosek.dinfitem.optimizer_time,
            ),
            s.NUM_ITERS: (
                task.getintinf(mosek.iinfitem.intpnt_iter)
                + task.getintinf(mosek.iinfitem.sim_primal_iter)
                + task.getintinf(mosek.iinfitem.sim_dual_iter)
                + task.getintinf(mosek.iinfitem.mio_num_relax)
            ),
            s.EXTRA_STATS: {
                "mio_intpnt_iter": task.getlintinf(
                    mosek.liinfitem.mio_intpnt_iter,
                ),
                "mio_simplex_iter": task.getlintinf(
                    mosek.liinfitem.mio_simplex_iter,
                ),
            },
        }

        # Determine status
        if (
            sol_type == mosek.soltype.itg
            and problem_status == mosek.prosta.prim_infeas
        ):
            status = s.INFEASIBLE
        elif problem_status == mosek.prosta.dual_infeas:
            status = s.UNBOUNDED
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]

        if status in s.SOLUTION_PRESENT:
            # Primal variables
            primal = np.array(task.getxx(sol_type))
            opt_val = (
                task.getprimalobj(sol_type)
                + inverse_data[s.OFFSET]
            )
            primal_vars = {
                inverse_data[self.VAR_ID]: primal,
            }

            # Dual variables
            if task.getnumintvar() > 0:
                dual_vars = {}
            else:
                num_con = task.getnumcon()
                if num_con > 0:
                    y = np.array(task.gety(sol_type))
                else:
                    y = np.array([])

                eq_dual = y[:cone_dims.zero]
                nonneg_dual = y[cone_dims.zero:]

                # ACC duals (SOC, RSOC, PSD, exp, pow)
                acc_duals = []
                num_acc = task.getnumacc()
                for i in range(num_acc):
                    doty = np.array(
                        task.getaccdoty(sol_type, i),
                    )
                    acc_duals.append(doty)

                if acc_duals:
                    ineq_dual = np.concatenate(
                        [nonneg_dual] + acc_duals,
                    )
                else:
                    ineq_dual = nonneg_dual

                eq_dual_vars = utilities.get_dual_values(
                    eq_dual,
                    utilities.extract_dual_value,
                    inverse_data[self.EQ_CONSTR],
                )
                ineq_dual_vars = utilities.get_dual_values(
                    ineq_dual,
                    self.extract_dual_value,
                    inverse_data[self.NEQ_CONSTR],
                )
                dual_vars = {}
                dual_vars.update(eq_dual_vars)
                dual_vars.update(ineq_dual_vars)

            sol = Solution(
                status, opt_val, primal_vars, dual_vars, attr,
            )
        else:
            sol = failure_solution(status, attr)

        task.__exit__(None, None, None)
        return sol

    # ------------------------------------------------------------------
    # extract_dual_value — ACC path dual extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extract dual values for the ACC path.

        Handles ExpCone (permuted order) and delegates to the
        default for everything else.
        """
        if isinstance(constraint, ExpCone):
            n_cones = constraint.num_cones()
            size = 3 * n_cones
            new_offset = offset + size
            dual = result_vec[offset:new_offset].copy()
            perm = expcone_permutor(
                n_cones, MOSEK.EXP_CONE_ORDER,
            )
            return dual[perm], new_offset
        else:
            return utilities.extract_dual_value(
                result_vec, offset, constraint,
            )

    # ------------------------------------------------------------------
    # Helpers for the dualized path
    # ------------------------------------------------------------------

    @staticmethod
    def recover_dual_variables(task, sol):
        """Recover dual variables for the dualized path."""
        if task.getnumintvar() == 0:
            dual_vars = dict()
            dual_var = [0.] * task.getnumcon()
            task.gety(sol, dual_var)
            dual_vars[s.EQ_DUAL] = np.array(dual_var)
        else:
            dual_vars = None
        return dual_vars

    @staticmethod
    def recover_primal_variables(task, sol, K_dir):
        """Recover primal variables for the dualized path."""
        prim_vars = dict()
        idx = 0
        m_free = K_dir[a2d.FREE]
        if m_free > 0:
            temp = [0.] * m_free
            task.getxxslice(sol, idx, len(temp), temp)
            prim_vars[a2d.FREE] = np.array(temp)
            idx += m_free
        if task.getnumintvar() > 0:
            return prim_vars
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
        num_rsoc = len(K_dir[a2d.RSOC])
        if num_rsoc > 0:
            rsoc_vars = []
            for dim in K_dir[a2d.RSOC]:
                temp = [0.] * dim
                task.getxxslice(sol, idx, idx + dim, temp)
                rsoc_vars.append(np.array(temp))
                idx += dim
            prim_vars[a2d.RSOC] = rsoc_vars
        num_dexp = K_dir[a2d.DUAL_EXP]
        if num_dexp > 0:
            temp = [0.] * (3 * num_dexp)
            task.getxxslice(sol, idx, idx + len(temp), temp)
            temp = np.array(temp)
            perm = expcone_permutor(
                num_dexp, MOSEK.EXP_CONE_ORDER,
            )
            prim_vars[a2d.DUAL_EXP] = temp[perm]
            idx += 3 * num_dexp
        num_dpow = len(K_dir[a2d.DUAL_POW3D])
        if num_dpow > 0:
            temp = [0.] * (3 * num_dpow)
            task.getxxslice(sol, idx, idx + len(temp), temp)
            temp = np.array(temp)
            prim_vars[a2d.DUAL_POW3D] = temp
            idx += 3 * num_dpow
        num_psd = len(K_dir[a2d.PSD])
        if num_psd > 0:
            psd_vars = []
            for j, dim in enumerate(K_dir[a2d.PSD]):
                xj = [0.] * (dim * (dim + 1) // 2)
                task.getbarxj(sol, j, xj)
                psd_vars.append(np.array(xj))
            prim_vars[a2d.PSD] = psd_vars
        return prim_vars

    # ------------------------------------------------------------------
    # Options handling
    # ------------------------------------------------------------------

    @staticmethod
    def handle_options(task, verbose: bool, solver_opts: dict) -> dict:
        """Handle user-specified solver options.

        Options that have to be applied before the optimization are
        applied to the task here.  A new dictionary is returned with
        the processed options and default options applied.
        """
        import mosek

        if verbose:
            def streamprinter(text):
                s.LOGGER.info(text.rstrip('\n'))

            print('\n')
            task.set_Stream(mosek.streamtype.log, streamprinter)

        solver_opts = MOSEK.parse_eps_keyword(solver_opts)

        mosek_params = solver_opts.pop('mosek_params', dict())
        if any(MOSEK.is_param(p) for p in mosek_params):
            warn(
                __MSK_ENUM_PARAM_DEPRECATION__,
                CvxpyDeprecationWarning,
            )
            warn(__MSK_ENUM_PARAM_DEPRECATION__)
        for param, value in mosek_params.items():
            if isinstance(param, str):
                param = param.strip()
                if isinstance(value, str):
                    task.putparam(param, value)
                else:
                    if param.startswith("MSK_DPAR_"):
                        task.putnadouparam(param, value)
                    elif param.startswith("MSK_IPAR_"):
                        task.putnaintparam(param, value)
                    elif param.startswith("MSK_SPAR_"):
                        task.putnastrparam(param, value)
                    else:
                        raise ValueError(
                            "Invalid MOSEK parameter "
                            "'%s'." % param
                        )
            else:
                if isinstance(param, mosek.dparam):
                    task.putdouparam(param, value)
                elif isinstance(param, mosek.iparam):
                    task.putintparam(param, value)
                elif isinstance(param, mosek.sparam):
                    task.putstrparam(param, value)
                else:
                    raise ValueError(
                        "Invalid MOSEK parameter "
                        "'%s'." % param
                    )

        processed_opts = dict()
        processed_opts['mosek_params'] = mosek_params
        processed_opts['save_file'] = solver_opts.pop(
            'save_file', False,
        )
        processed_opts['bfs'] = solver_opts.pop('bfs', False)
        processed_opts['accept_unknown'] = solver_opts.pop(
            'accept_unknown', False,
        )

        if solver_opts:
            raise ValueError(
                f"Invalid keyword-argument(s) "
                f"{solver_opts.keys()} passed to MOSEK solver."
            )

        if processed_opts['bfs']:
            task.putintparam(
                mosek.iparam.intpnt_basis,
                mosek.basindtype.always,
            )
        else:
            task.putintparam(
                mosek.iparam.intpnt_basis,
                mosek.basindtype.never,
            )
        return processed_opts

    @staticmethod
    def is_param(param: object) -> bool:
        import mosek
        return isinstance(
            param, (mosek.iparam, mosek.dparam, mosek.sparam),
        )

    @staticmethod
    def parse_eps_keyword(solver_opts: dict) -> dict:
        """Parse the eps keyword and update the corresponding MOSEK
        parameters.  If additional tolerances are specified explicitly,
        they take precedence over the eps keyword.
        """
        if 'eps' not in solver_opts:
            return solver_opts

        tol_params = MOSEK.tolerance_params()
        mosek_params = solver_opts.get('mosek_params', dict())
        assert not any(
            MOSEK.is_param(p) for p in mosek_params
        ), (
            "The eps keyword is not compatible with "
            "(deprecated) Mosek enum parameters. "
            "Use the string parameters instead."
        )
        solver_opts['mosek_params'] = mosek_params
        eps = solver_opts.pop('eps')
        for tol_param in tol_params:
            solver_opts['mosek_params'][tol_param] = (
                solver_opts['mosek_params'].get(tol_param, eps)
            )
        return solver_opts

    @staticmethod
    def tolerance_params() -> tuple[str]:
        return (
            # Conic interior-point tolerances
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS",
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED",
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            # Interior-point tolerances
            "MSK_DPAR_INTPNT_TOL_DFEAS",
            "MSK_DPAR_INTPNT_TOL_INFEAS",
            "MSK_DPAR_INTPNT_TOL_MU_RED",
            "MSK_DPAR_INTPNT_TOL_PFEAS",
            "MSK_DPAR_INTPNT_TOL_REL_GAP",
            # Simplex tolerances
            "MSK_DPAR_BASIS_REL_TOL_S",
            "MSK_DPAR_BASIS_TOL_S",
            "MSK_DPAR_BASIS_TOL_X",
            # MIO tolerances
            "MSK_DPAR_MIO_TOL_ABS_GAP",
            "MSK_DPAR_MIO_TOL_ABS_RELAX_INT",
            "MSK_DPAR_MIO_TOL_FEAS",
            "MSK_DPAR_MIO_TOL_REL_GAP",
        )

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        citation = CITATION_DICT['MOSEK']

        # We need another citation if nonsymmetric cones are present.
        nonsym_cones = False
        # Dualized path stores dual cone info in K_dir.
        K_dir = data.get('K_dir', dict())
        if K_dir.get(DUAL_EXP, 0):
            nonsym_cones = True
        if K_dir.get(DUAL_POW3D, 0):
            nonsym_cones = True
        # ACC path stores cone info in cone_dims.
        cone_dims = data.get(ConicSolver.DIMS, None)
        if cone_dims is not None:
            if getattr(cone_dims, 'exp', 0) > 0:
                nonsym_cones = True
            if getattr(cone_dims, 'p3d', None):
                nonsym_cones = True
            if getattr(cone_dims, 'pnd', None):
                nonsym_cones = True

        if nonsym_cones:
            citation += CITATION_DICT['MOSEK_EXP']
        return citation
