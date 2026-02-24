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

from collections import defaultdict

import numpy as np
import scipy as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D, PowConeND
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.warn import CvxpyDeprecationWarning, warn

__MSK_ENUM_PARAM_DEPRECATION__ = """
Using MOSEK constants to specify parameters is deprecated.
Use generic string names instead.
For example, replace mosek.iparam.num_threads with 'MSK_IPAR_NUM_THREADS'
"""


def svec_to_full_mat(svec, n):
    """Expands n*(n+1)//2 svec to full symmetric matrix.

    Unscales off-diagonal by 1/sqrt(2), as per the svec convention
    used by MOSEK's appendsvecpsdconedomain.

    Parameters
    ----------
    svec : numpy.ndarray
        svec representation (lower triangular, column-major, sqrt(2)-scaled off-diag).
    n : int
        Matrix dimension.

    Returns
    -------
    numpy.ndarray
        Flattened full symmetric matrix (column-major, n*n elements).

    Notes
    -----
    The column-major lower triangular ordering used by MOSEK's svec matches
    numpy's upper triangular indices. This is because (i, j) in column-major
    lower triangular corresponds to (j, i) in row-major upper triangular,
    and the matrix is symmetric so X[i,j] = X[j,i].
    """
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = svec
    full += full.T
    full[np.diag_indices(n)] /= 2
    full[np.tril_indices(n, k=-1)] /= np.sqrt(2)
    full[np.triu_indices(n, k=1)] /= np.sqrt(2)
    return np.reshape(full, n * n, order="F")


class MOSEK(ConicSolver):
    """An interface for the MOSEK solver.

    Requires MOSEK 10 or newer. Uses the ACC (Affine Conic Constraints) API
    to express constraints directly in primal form without dualization.
    """

    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    PSD_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC, PSD, ExpCone, PowCone3D, PowConeND,
    ]
    EXP_CONE_ORDER = [2, 1, 0]
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [
        SOC, ExpCone, PowCone3D, PowConeND,
    ]

    """
    The cvxpy standard for the exponential cone is:
        K_e = closure{(x,y,z) |  z >= y * exp(x/y), y>0}.
    Whenever a solver uses this convention, EXP_CONE_ORDER should be [0, 1, 2].

    MOSEK uses the convention:
        K_e = closure{(x,y,z) | x >= y * exp(z/y), x,y >= 0}.
    with this convention, EXP_CONE_ORDER should be [2, 1, 0].
    """

    def import_solver(self) -> None:
        """Imports the solver and verifies MOSEK 10+."""
        import mosek  # noqa F401
        if not hasattr(mosek.Task, 'appendafes'):
            raise RuntimeError("CVXPY requires MOSEK 10.0 or newer.")

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def accepts(self, problem) -> bool:
        """Can the installed version of Mosek solve the problem?
        """
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
        Off-diagonal entries are scaled by sqrt(2) to match MOSEK's
        ``appendsvecpsdconedomain`` convention (same as SCS).
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1)//2

        row_arr = np.arange(0, entries)

        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows*cols)
        scaled_lower_tri = sp.sparse.csc_array((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.sparse.csc_array((val_symm, (row_symm, col_symm)))

        return scaled_lower_tri @ symm_matrix

    def apply(self, problem):
        """Returns problem data and inverse data for the MOSEK ACC API.

        Uses the base class ``_prepare_data_and_inv_data`` to format constraints,
        then extracts the problem data (A, b, c) directly (no dualization).
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)
        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        data[s.A] = A  # NOT negated (ACC uses Ax + b directly)
        data[s.B] = b
        inv_data[s.OFFSET] = d
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds
        data[s.BOOL_IDX] = [int(t[0]) for t in problem.x.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in problem.x.integer_idx]
        data['psd_variable_info'] = problem.psd_variable_info
        inv_data['psd_variable_info'] = problem.psd_variable_info
        return data, inv_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import mosek

        task = mosek.Task()
        solver_opts = MOSEK.handle_options(task, verbose, solver_opts)
        MOSEK._build_task(task, data)

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

    @staticmethod
    def _sparse_coo(A_block):
        """Extract (row, col, val) COO entries from a dense or sparse matrix."""
        if sp.sparse.issparse(A_block):
            A_coo = A_block.tocoo()
            return A_coo.row, A_coo.col, A_coo.data
        else:
            rows, cols = np.nonzero(A_block)
            vals = np.asarray(A_block[rows, cols]).ravel()
            return rows, cols, vals

    @staticmethod
    def _add_afe_acc(task, A, b, row, dim, domain_idx):
        """Add an ACC for rows A[row:row+dim, :] x + b[row:row+dim] in domain.

        Returns the AFE base index.
        """
        afe_base = task.getnumafe()
        task.appendafes(dim)
        rows_i, cols_j, vals = MOSEK._sparse_coo(A[row:row + dim, :])
        if len(vals) > 0:
            task.putafefentrylist(
                (rows_i + afe_base).tolist(),
                cols_j.tolist(),
                vals.tolist(),
            )
        task.putafegslice(afe_base, afe_base + dim, b[row:row + dim].tolist())
        task.appendacc(domain_idx, list(range(afe_base, afe_base + dim)), None)
        return afe_base

    @staticmethod
    def _build_task(task, data):
        """Build a MOSEK task using ACC API for all conic constraints.

        Linear (zero/nonneg) constraints use efficient MOSEK linear constraints.
        All other cones (SOC, PSD, exp, pow3d, powND) use ACC.

        PSD variables are handled without linking constraints: barvar coefficients
        are injected directly into the objective, linear constraints, and AFE rows
        via putbarcblocktriplet / putbarablocktriplet / putafebarfblocktriplet.
        """
        import mosek

        c = data[s.C]
        A = data[s.A]
        b = data[s.B]
        cone_dims = data[ConicSolver.DIMS]
        n = c.size

        # Variables
        task.appendvars(n)

        # Create barvars for PSD variables early.
        psd_var_info = data.get('psd_variable_info', [])
        psd_var_barvars = []
        for var_offset, reduced_size, psd_dim, is_nsd in psd_var_info:
            barvar_idx = task.getnumbarvar()
            task.appendbarvars([psd_dim])
            # PSD variable reduced form uses upper triangular row-major ordering
            # (from upper_tri_to_full). MOSEK needs lower triangular (k >= l),
            # so we swap row/col.
            tri_rows, tri_cols = np.triu_indices(psd_dim)
            psd_var_barvars.append((
                var_offset, reduced_size, psd_dim, is_nsd,
                barvar_idx, tri_cols, tri_rows,
            ))

        # Variable bounds
        if data[s.LOWER_BOUNDS] is not None and data[s.UPPER_BOUNDS] is not None:
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
            task.putvarboundlist(np.arange(n, dtype=np.int32), list(bk), bl, bu)
        else:
            o = np.zeros(n)
            task.putvarboundlist(np.arange(n, dtype=np.int32),
                                [mosek.boundkey.fr] * n, o, o)

        # Fix PSD variable scalar entries to zero.
        # Their A-matrix coefficients contribute nothing; barvar coefficients
        # (added later) provide the actual contribution.
        for var_offset, reduced_size, *_ in psd_var_barvars:
            idx = np.arange(var_offset, var_offset + reduced_size,
                            dtype=np.int32)
            task.putvarboundlist(idx, [mosek.boundkey.fx] * reduced_size,
                                [0.0] * reduced_size, [0.0] * reduced_size)

        # Objective: min c'x
        task.putclist(np.arange(n, dtype=np.int32), c)
        task.putobjsense(mosek.objsense.minimize)

        # Track A-row to MOSEK-index mappings for PSD variable post-processing.
        linear_maps = []  # (A_row_start, A_row_end, mosek_con_offset)
        afe_maps = []     # (A_row_start, A_row_end, mosek_afe_base)

        row = 0

        # Zero cone -> MOSEK linear equality constraints
        # Ax + b = 0  =>  Ax = -b  (fixed bound)
        num_eq = cone_dims.zero
        if num_eq > 0:
            con_offset = task.getnumcon()
            task.appendcons(num_eq)
            eq_rows, eq_cols, eq_vals = MOSEK._sparse_coo(A[row:row + num_eq, :])
            if len(eq_vals) > 0:
                task.putaijlist(
                    (eq_rows + con_offset).tolist(),
                    eq_cols.tolist(),
                    eq_vals.tolist(),
                )
            bounds = (-b[row:row + num_eq]).tolist()
            task.putconboundlist(
                np.arange(con_offset, con_offset + num_eq, dtype=np.int32),
                [mosek.boundkey.fx] * num_eq,
                bounds, bounds,
            )
            linear_maps.append((row, row + num_eq, con_offset))
            row += num_eq

        # NonNeg cone -> MOSEK linear inequality constraints
        # Ax + b >= 0  =>  Ax >= -b  (lower bound)
        num_nn = cone_dims.nonneg
        if num_nn > 0:
            con_offset = task.getnumcon()
            task.appendcons(num_nn)
            nn_rows, nn_cols, nn_vals = MOSEK._sparse_coo(A[row:row + num_nn, :])
            if len(nn_vals) > 0:
                task.putaijlist(
                    (nn_rows + con_offset).tolist(),
                    nn_cols.tolist(),
                    nn_vals.tolist(),
                )
            lb = (-b[row:row + num_nn]).tolist()
            ub = [0.0] * num_nn
            task.putconboundlist(
                np.arange(con_offset, con_offset + num_nn, dtype=np.int32),
                [mosek.boundkey.lo] * num_nn,
                lb, ub,
            )
            linear_maps.append((row, row + num_nn, con_offset))
            row += num_nn

        # SOC cones via ACC
        for dim in cone_dims.soc:
            afe_base = MOSEK._add_afe_acc(
                task, A, b, row, dim, task.appendquadraticconedomain(dim))
            afe_maps.append((row, row + dim, afe_base))
            row += dim

        # PSD cones via ACC with svec PSD domain
        for psd_dim in cone_dims.psd:
            vec_len = psd_dim * (psd_dim + 1) // 2
            afe_base = MOSEK._add_afe_acc(
                task, A, b, row, vec_len,
                task.appendsvecpsdconedomain(vec_len))
            afe_maps.append((row, row + vec_len, afe_base))
            row += vec_len

        # Exp cones via ACC
        for _ in range(cone_dims.exp):
            afe_base = MOSEK._add_afe_acc(
                task, A, b, row, 3, task.appendprimalexpconedomain())
            afe_maps.append((row, row + 3, afe_base))
            row += 3

        # Pow3D cones via ACC
        for alpha in cone_dims.p3d:
            dom = task.appendprimalpowerconedomain(3, [alpha, 1.0 - alpha])
            afe_base = MOSEK._add_afe_acc(task, A, b, row, 3, dom)
            afe_maps.append((row, row + 3, afe_base))
            row += 3

        # PowND cones via ACC
        for alpha in cone_dims.pnd:
            dim = len(alpha) + 1
            dom = task.appendprimalpowerconedomain(dim, list(alpha))
            afe_base = MOSEK._add_afe_acc(task, A, b, row, dim, dom)
            afe_maps.append((row, row + dim, afe_base))
            row += dim

        # Add barvar coefficients for PSD variables into
        # objective, linear constraints, and AFE rows.
        MOSEK._add_psd_var_barvar_coefficients(
            task, c, A, psd_var_barvars, linear_maps, afe_maps)

        # Integer constraints
        bool_idx = data[s.BOOL_IDX]
        int_idx = data[s.INT_IDX]
        num_bool = len(bool_idx)
        num_int = len(int_idx)
        if num_bool + num_int > 0:
            vartypes = [mosek.variabletype.type_int] * (num_bool + num_int)
            task.putvartypelist(int_idx + bool_idx, vartypes)
        if num_bool > 0:
            task.putvarboundlist(bool_idx, [mosek.boundkey.ra] * num_bool,
                                [0] * num_bool, [1] * num_bool)

        return task

    @staticmethod
    def _psd_var_barvar_triplets(A, row_start, row_end, idx_offset,
                                 psd_var_barvars):
        """Compute barvar block triplet entries for A[row_start:row_end, psd_cols].

        For each PSD variable barvar, extracts nonzero entries from the
        corresponding columns of A and converts them to MOSEK's sparse
        symmetric matrix format for putbarablocktriplet / putafebarfblocktriplet.

        Returns (indices, barvar_indices, subk, subl, vals).
        """
        all_idx, all_bv, all_k, all_l, all_v = [], [], [], [], []
        for (var_offset, reduced_size, psd_dim, is_nsd,
             barvar_idx, tri_rows, tri_cols) in psd_var_barvars:
            row_indices, col_indices, a_vals = MOSEK._sparse_coo(
                A[row_start:row_end, var_offset:var_offset + reduced_size])
            if len(a_vals) == 0:
                continue

            sign = -1.0 if is_nsd else 1.0
            i_k = tri_rows[col_indices]
            j_k = tri_cols[col_indices]
            # Diagonal: val=1.0 so <M,X> = X[i,i].
            # Off-diagonal: val=0.5 so <M,X> = 2*0.5*X[i,j] = X[i,j].
            scale = np.where(i_k == j_k, 1.0, 0.5)

            all_idx.extend((row_indices + idx_offset).tolist())
            all_bv.extend([barvar_idx] * len(row_indices))
            all_k.extend(i_k.astype(np.int32).tolist())
            all_l.extend(j_k.astype(np.int32).tolist())
            all_v.extend((sign * a_vals * scale).tolist())
        return all_idx, all_bv, all_k, all_l, all_v

    @staticmethod
    def _add_psd_var_barvar_coefficients(task, c, A, psd_var_barvars,
                                         linear_maps, afe_maps):
        """Add barvar coefficients for PSD variables to objective, constraints, AFEs.

        Instead of linking constraints (v_k = <M_k, X>), we inject barvar
        coefficients directly into the rows where the scalar PSD variable
        entries appear, achieving the same effect with zero extra constraints.
        """
        # Objective: putbarcblocktriplet
        all_j, all_k, all_l, all_v = [], [], [], []
        for (var_offset, reduced_size, psd_dim, is_nsd,
             barvar_idx, tri_rows, tri_cols) in psd_var_barvars:
            c_psd = c[var_offset:var_offset + reduced_size]
            nz = np.nonzero(c_psd)[0]
            if len(nz) == 0:
                continue
            sign = -1.0 if is_nsd else 1.0
            i_k, j_k = tri_rows[nz], tri_cols[nz]
            scale = np.where(i_k == j_k, 1.0, 0.5)
            all_j.extend([barvar_idx] * len(nz))
            all_k.extend(i_k.astype(np.int32).tolist())
            all_l.extend(j_k.astype(np.int32).tolist())
            all_v.extend((sign * c_psd[nz] * scale).tolist())
        if all_j:
            task.putbarcblocktriplet(all_j, all_k, all_l, all_v)

        # Linear constraints: putbarablocktriplet
        all_subi, all_subj, all_subk, all_subl, all_vals = (
            [], [], [], [], []
        )
        for row_start, row_end, con_offset in linear_maps:
            t = MOSEK._psd_var_barvar_triplets(
                A, row_start, row_end, con_offset, psd_var_barvars)
            all_subi.extend(t[0])
            all_subj.extend(t[1])
            all_subk.extend(t[2])
            all_subl.extend(t[3])
            all_vals.extend(t[4])
        if all_subi:
            task.putbarablocktriplet(
                all_subi, all_subj, all_subk, all_subl, all_vals)

        # AFE constraints: putafebarfblocktriplet
        all_afe, all_bv, all_k, all_l, all_v = [], [], [], [], []
        for row_start, row_end, afe_base in afe_maps:
            t = MOSEK._psd_var_barvar_triplets(
                A, row_start, row_end, afe_base, psd_var_barvars)
            all_afe.extend(t[0])
            all_bv.extend(t[1])
            all_k.extend(t[2])
            all_l.extend(t[3])
            all_v.extend(t[4])
        if all_afe:
            task.putafebarfblocktriplet(
                all_afe, all_bv, all_k, all_l, all_v)

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extract dual value for a constraint.

        Special-cases PSD constraints (expand svec to full matrix) and
        ExpCone constraints (un-permute from MOSEK's ordering).
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = svec_to_full_mat(lower_tri, dim)
            return full, new_offset
        elif isinstance(constraint, ExpCone):
            n_cones = constraint.num_cones()
            size = 3 * n_cones
            new_offset = offset + size
            dual = result_vec[offset:new_offset].copy()
            perm = expcone_permutor(n_cones, MOSEK.EXP_CONE_ORDER)
            return dual[perm], new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def _extract_dual_vars(self, task, sol_type, cone_dims, inverse_data):
        """Extract dual variables from a MOSEK task.

        Used for both optimal solutions and infeasibility certificates (IIS).

        Returns
        -------
        dict
            Mapping from constraint id to numpy array of dual values.
        """
        num_con = task.getnumcon()
        if num_con > 0:
            y = np.array(task.gety(sol_type))
        else:
            y = np.array([])

        eq_dual = y[:cone_dims.zero]
        nonneg_dual = y[cone_dims.zero:
                        cone_dims.zero + cone_dims.nonneg]

        # ACC duals in creation order: SOC, PSD, exp, pow3d, powND
        # This matches NEQ_CONSTR order (after nonneg).
        acc_duals = []
        for i in range(task.getnumacc()):
            acc_duals.append(np.array(task.getaccdoty(sol_type, i)))

        ineq_dual = np.concatenate([nonneg_dual] + acc_duals)

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
        return dual_vars

    def invert(self, solver_output, inverse_data):
        """Map MOSEK solution back to CVXPY's standard form.

        Extracts primal variables via ``getxx``, duals via ``gety``
        (for zero/nonneg linear constraints), and ACC duals via
        ``getaccdoty`` (for SOC/PSD/exp/pow constraints).
        """
        import mosek

        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.integer_optimal: s.OPTIMAL,
                      mosek.solsta.prim_feas: s.OPTIMAL_INACCURATE,
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED}

        task = solver_output['task']
        solver_opts = solver_output['solver_options']

        if solver_opts['accept_unknown']:
            STATUS_MAP[mosek.solsta.unknown] = s.OPTIMAL_INACCURATE

        STATUS_MAP = defaultdict(lambda: s.SOLVER_ERROR, STATUS_MAP)

        cone_dims = inverse_data[self.DIMS]

        # Determine solution type
        simplex_algs = [
            mosek.optimizertype.primal_simplex,
            mosek.optimizertype.dual_simplex,
        ]
        current_optimizer = task.getintparam(mosek.iparam.optimizer)
        bfs_active = (solver_opts.get("bfs", False)
                      and task.getnumacc() == 0
                      and task.getnumbarvar() == 0)

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif current_optimizer in simplex_algs or bfs_active:
            sol_type = mosek.soltype.bas
        else:
            sol_type = mosek.soltype.itr

        # Solver attributes
        problem_status = task.getprosta(sol_type)
        attr = {s.SOLVE_TIME: task.getdouinf(mosek.dinfitem.optimizer_time),
                s.NUM_ITERS: task.getintinf(mosek.iinfitem.intpnt_iter) +
                             task.getintinf(mosek.iinfitem.sim_primal_iter) +
                             task.getintinf(mosek.iinfitem.sim_dual_iter) +
                             task.getintinf(mosek.iinfitem.mio_num_relax),
                s.EXTRA_STATS: {
                    "mio_intpnt_iter": task.getlintinf(
                        mosek.liinfitem.mio_intpnt_iter),
                    "mio_simplex_iter": task.getlintinf(
                        mosek.liinfitem.mio_simplex_iter),
                }
        }

        # Determine status
        if sol_type == mosek.soltype.itg and problem_status == mosek.prosta.prim_infeas:
            status = s.INFEASIBLE
        elif problem_status == mosek.prosta.dual_infeas:
            status = s.UNBOUNDED
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]

        if status in s.SOLUTION_PRESENT:
            # Primal variables
            primal = np.array(task.getxx(sol_type))
            opt_val = task.getprimalobj(sol_type) + inverse_data[s.OFFSET]

            # Extract barvar values for PSD variables and populate primal.
            # Scalar PSD var entries are fixed to 0 in the task; the actual
            # values live in the barvars. MOSEK's getbarxj returns lower
            # triangular column-major entries, which for symmetric matrices
            # are identical to upper triangular row-major entries used by
            # CVXPY's reduced PSD variable representation.
            psd_var_info = inverse_data.get('psd_variable_info', [])
            for barvar_idx, (var_offset, reduced_size, psd_dim, is_nsd) \
                    in enumerate(psd_var_info):
                barxj = [0.0] * reduced_size
                task.getbarxj(sol_type, barvar_idx, barxj)
                sign = -1.0 if is_nsd else 1.0
                primal[var_offset:var_offset + reduced_size] = (
                    sign * np.array(barxj))

            primal_vars = {inverse_data[self.VAR_ID]: primal}

            # Dual variables
            if task.getnumintvar() > 0:
                dual_vars = None
            else:
                dual_vars = self._extract_dual_vars(
                    task, sol_type, cone_dims, inverse_data
                )

            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            # Extract IIS (Farkas certificate) for infeasible continuous problems.
            # MOSEK's dual variables for solsta.prim_infeas_cer are the Farkas
            # certificate — non-zero entries indicate participating constraints.
            if status == s.INFEASIBLE and task.getnumintvar() == 0:
                try:
                    iis = self._extract_dual_vars(
                        task, sol_type, cone_dims, inverse_data
                    )
                    attr[s.EXTRA_STATS]["IIS"] = iis
                except Exception:
                    pass
            sol = failure_solution(status, attr)

        # Delete the mosek Task and Environment
        task.__exit__(None, None, None)

        return sol

    @staticmethod
    def handle_options(task, verbose: bool, solver_opts: dict) -> dict:
        """
        Handle user-specified solver options.

        Options that have to be applied before the optimization are applied to the task here.
        A new dictionary is returned with the processed options and default options applied.
        """

        # If verbose, then set default logging parameters.
        import mosek

        if verbose:

            def streamprinter(text):
                s.LOGGER.info(text.rstrip('\n'))

            print('\n')

            task.set_Stream(mosek.streamtype.log, streamprinter)

        solver_opts = MOSEK.parse_eps_keyword(solver_opts)

        # Parse all user-specified parameters (override default logging
        # parameters if applicable).

        mosek_params = solver_opts.pop('mosek_params', dict())
        # Issue a warning if Mosek enums are used as parameter names / keys
        if any(MOSEK.is_param(p) for p in mosek_params):
            warn(__MSK_ENUM_PARAM_DEPRECATION__, CvxpyDeprecationWarning)
            warn(__MSK_ENUM_PARAM_DEPRECATION__)
        # Now set parameters
        for param, value in mosek_params.items():
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

        # The options as passed to the solver
        processed_opts = dict()
        processed_opts['mosek_params'] = mosek_params
        processed_opts['save_file'] = solver_opts.pop('save_file', False)
        processed_opts['bfs'] = solver_opts.pop('bfs', False)
        processed_opts['accept_unknown'] = solver_opts.pop('accept_unknown', False)

        # Check if any unknown options were passed
        if solver_opts:
            raise ValueError(f"Invalid keyword-argument(s) {solver_opts.keys()} passed "
                             f"to MOSEK solver.")

        # Decide whether basis identification is needed for intpnt solver
        # This is only required if solve() was called with bfs=True
        if processed_opts['bfs']:
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.always)
        else:
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        return processed_opts

    @staticmethod
    def is_param(param: str | "iparam" | "dparam" | "sparam") -> bool:  # noqa: F821
        import mosek
        return isinstance(param, (mosek.iparam, mosek.dparam,  mosek.sparam))

    @staticmethod
    def parse_eps_keyword(solver_opts: dict) -> dict:
        """
        Parse the eps keyword and update the corresponding MOSEK parameters.
        If additional tolerances are specified explicitly, they take precedence over the
        eps keyword.
        """

        if 'eps' not in solver_opts:
            return solver_opts

        tol_params = MOSEK.tolerance_params()
        mosek_params = solver_opts.get('mosek_params', dict())
        assert not any(MOSEK.is_param(p) for p in mosek_params), \
            "The eps keyword is not compatible with (deprecated) Mosek enum parameters. \
            Use the string parameters instead."
        solver_opts['mosek_params'] = mosek_params
        eps = solver_opts.pop('eps')
        for tol_param in tol_params:
            solver_opts['mosek_params'][tol_param] = \
                solver_opts['mosek_params'].get(tol_param, eps)
        return solver_opts

    @staticmethod
    def tolerance_params() -> tuple[str]:
        # tolerance parameters from
        # https://docs.mosek.com/latest/pythonapi/param-groups.html
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
            "MSK_DPAR_MIO_TOL_REL_GAP"
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
        cone_dims = data.get(self.DIMS, None)
        if cone_dims is not None:
            if (cone_dims.exp > 0 or len(cone_dims.p3d) > 0
                    or len(cone_dims.pnd) > 0):
                citation += CITATION_DICT['MOSEK_EXP']
        return citation
