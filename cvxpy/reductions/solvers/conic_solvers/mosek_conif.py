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

import numpy as np
import scipy as sp
from cvxpy.reductions.solvers.utilities import expcone_permutor
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, NonNeg, Zero, ExpCone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from collections import defaultdict


def vectorized_lower_tri_to_mat(v, dim):
    """
    :param v: a list of length (dim * (dim + 1) / 2)
    :param dim: the number of rows (equivalently, columns) in the output array.
    :return: Return the symmetric 2D array defined by taking "v" to
      specify its lower triangular entries.
    """
    rows, cols, vals = [], [], []
    running_idx = 0
    for j in range(dim):
        rows += [j + k for k in range(dim - j)]
        cols += [j] * (dim - j)
        vals += v[running_idx:(running_idx + dim - j)]
        running_idx += dim - j
    A = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).toarray()
    d = np.diag(np.diag(A))
    A = A + A.T - d
    return A


class MOSEK(ConicSolver):
    """ An interface for the Mosek solver.
    """

    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, PSD]
    EXP_CONE_ORDER = [2, 1, 0]
    DUAL_EXP_CONE_ORDER = [0, 1, 2]

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

    def import_solver(self):
        """Imports the solver (updates the set of supported constraints, if applicable).
        """
        import mosek
        mosek  # For flake8
        if hasattr(mosek.conetype, 'pexp') and ExpCone not in MOSEK.SUPPORTED_CONSTRAINTS:
            MOSEK.SUPPORTED_CONSTRAINTS.append(ExpCone)

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def accepts(self, problem):
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

    def old_apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = dict()
        var = problem.x
        inv_data = {self.VAR_ID: var.id,
                    'suc_slacks': [], 'y_slacks': [], 'snx_slacks': [], 'psd_dims': []}

        # Get integrality constraint information
        data[s.BOOL_IDX] = [int(t[0]) for t in var.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in var.integer_idx]
        inv_data['integer_variables'] = len(data[s.BOOL_IDX]) + len(data[s.INT_IDX]) > 0

        if not problem.formatted:
            problem = self.format_constraints(problem,
                                              MOSEK.EXP_CONE_ORDER)
        data[s.PARAM_PROB] = problem
        constr_map = problem.constr_map
        data[s.DIMS] = problem.cone_dims

        inv_data['constraints'] = problem.constraints

        # A is ordered as [Zero, NonNeg, SOC, PSD, EXP]
        c, d, A, b = problem.apply_parameters()
        A = -A
        data[s.C] = c.ravel()
        inv_data['n0'] = len(data[s.C])
        data[s.OBJ_OFFSET] = float(d)
        inv_data[s.OBJ_OFFSET] = float(d)

        Gs = []
        hs = []
        # Linear inequalities
        num_linear_equalities = len(constr_map[Zero])
        num_linear_inequalities = len(constr_map[NonNeg])
        leq_dim = data[s.DIMS][s.LEQ_DIM]
        eq_dim = data[s.DIMS][s.EQ_DIM]
        if num_linear_inequalities > 0:
            # G, h : G * z <= h
            offset = num_linear_equalities
            for c in problem.constraints[offset:offset + num_linear_inequalities]:
                assert(isinstance(c, NonNeg))
                inv_data['suc_slacks'].append((c.id, c.size))
            row_offset = eq_dim
            Gs.append(A[row_offset:row_offset + leq_dim])
            hs.append(b[row_offset:row_offset + leq_dim])

        # Linear equations
        if num_linear_equalities > 0:
            for c in problem.constraints[:num_linear_equalities]:
                assert(isinstance(c, Zero))
                inv_data['y_slacks'].append((c.id, c.size))
            Gs.append(A[:eq_dim])
            hs.append(b[:eq_dim])

        # Second order cone
        num_soc = len(constr_map[SOC])
        soc_dim = sum(data[s.DIMS][s.SOC_DIM])
        if num_soc > 0:
            offset = num_linear_inequalities + num_linear_equalities
            for c in problem.constraints[offset:offset + num_soc]:
                assert(isinstance(c, SOC))
                inv_data['snx_slacks'].append((c.id, c.size))
            row_offset = leq_dim + eq_dim
            Gs.append(A[row_offset:row_offset + soc_dim])
            hs.append(b[row_offset:row_offset + soc_dim])

        # Exponential cone
        num_exp = len(constr_map[ExpCone])
        if num_exp > 0:
            # G * z <=_{EXP} h.
            len_exp = 0
            for c in problem.constraints[-num_exp:]:
                assert(isinstance(c, ExpCone))
                inv_data['snx_slacks'].append((c.id, 3 * c.num_cones()))
                len_exp += 3 * c.num_cones()
            Gs.append(A[-len_exp:])
            hs.append(b[-len_exp:])

        # PSD constraints
        num_psd = len(constr_map[PSD])
        psd_dim = sum([dim ** 2 for dim in data[s.DIMS][s.PSD_DIM]])
        if num_psd > 0:
            offset = num_linear_inequalities + num_linear_equalities + num_soc
            for c in problem.constraints[offset:offset + num_psd]:
                assert(isinstance(c, PSD))
                inv_data['psd_dims'].append((c.id, c.expr.shape[0]))
            row_offset = leq_dim + eq_dim + soc_dim
            Gs.append(A[row_offset:row_offset + psd_dim])
            hs.append(b[row_offset:row_offset + psd_dim])

        if Gs:
            data[s.G] = sp.sparse.vstack(tuple(Gs))
        else:
            data[s.G] = sp.sparse.csc_matrix((0, 0))
        if hs:
            data[s.H] = np.hstack(tuple(hs))
        else:
            data[s.H] = np.array([])
        inv_data['is_LP'] = (len(constr_map[PSD]) +
                             len(constr_map[ExpCone]) +
                             len(constr_map[SOC])) == 0

        return data, inv_data

    def old_solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import mosek
        env = mosek.Env()
        task = env.Task(0, 0)
        # If verbose, then set default logging parameters.
        if verbose:
            import sys

            def streamprinter(text):
                sys.stdout.write(text)
                sys.stdout.flush()
            print('\n')
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)
            task.putintparam(mosek.iparam.infeas_report_auto, mosek.onoffkey.on)
            task.putintparam(mosek.iparam.log_presolve, 0)

        # Parse all user-specified parameters (override default logging
        # parameters if applicable).
        kwargs = sorted(solver_opts.keys())
        save_file = None
        bfs = False
        if 'mosek_params' in kwargs:
            self._handle_mosek_params(task, solver_opts['mosek_params'])
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

        # Check if the cvxpy standard form has zero variables. If so,
        # return a trivial solution. This is necessary because MOSEK
        # will crash if handed a problem with zero variables.
        if len(data[s.C]) == 0:
            return {s.STATUS: s.OPTIMAL, s.PRIMAL: [],
                    s.VALUE: data[s.OFFSET], s.EQ_DUAL: [], s.INEQ_DUAL: []}

        # The following lines recover problem parameters, and define helper constants.
        #
        #   The problem's objective is "min c.T * z".
        #   The problem's constraint set is "G * z <=_K h."
        #   The rows in (G, h) are formatted in order of
        #       (1) linear inequalities,
        #       (2) linear equations,
        #       (3) soc constraints,
        #       (4) exponential cone constraints,
        #       (5) vectorized linear matrix inequalities.
        #   The parameter "dims" indicates the exact
        #   dimensions of each of these cones.
        #
        #   MOSEK's standard form requires that we replace generalized
        #   inequalities with slack variables and linear equations.
        #   The parameter "n" is the size of the column-vector variable
        #   after adding slacks for SOC and EXP constraints. To be
        #   consistent with MOSEK documentation, subsequent comments
        #   refer to this variable as "x".

        c = data[s.C]
        G, h = data[s.G], data[s.H]
        dims = data[s.DIMS]
        n0 = len(c)
        n = n0 + sum(dims[s.SOC_DIM]) + 3 * dims[s.EXP_DIM]
        psd_total_dims = sum(el ** 2 for el in dims[s.PSD_DIM])
        m = len(h)
        num_bool = len(data[s.BOOL_IDX])
        num_int = len(data[s.INT_IDX])

        # Define variables, cone constraints, and integrality constraints.
        #
        #   The variable "x" is a length-n block vector, with
        #       Block 1: "z" from "G * z <=_K h",
        #       Block 2: slacks for SOC constraints, and
        #       Block 3: slacks for EXP cone constraints.
        #
        #   Once we declare x in the MOSEK model, we add the necessary
        #   conic constraints for slack variables (Blocks 2 and 3).
        #   The last step is to add integrality constraints.
        #
        #   Note that the API call for PSD variables contains the word "bar".
        #   MOSEK documentation consistently uses "bar" as a sort of flag,
        #   indicating that a function deals with PSD variables.

        task.appendvars(n)
        task.putvarboundlist(np.arange(n, dtype=int),
                             [mosek.boundkey.fr] * n, np.zeros(n), np.zeros(n))
        if psd_total_dims > 0:
            task.appendbarvars(dims[s.PSD_DIM])
        running_idx = n0
        for size_cone in dims[s.SOC_DIM]:
            task.appendcone(mosek.conetype.quad,
                            0.0,  # unused
                            np.arange(running_idx, running_idx + size_cone))
            running_idx += size_cone
        for k in range(dims[s.EXP_DIM]):
            task.appendcone(mosek.conetype.pexp,
                            0.0,  # unused
                            np.arange(running_idx, running_idx + 3))
            running_idx += 3
        if num_bool + num_int > 0:
            task.putvartypelist(data[s.BOOL_IDX], [mosek.variabletype.type_int] * num_bool)
            task.putvarboundlist(data[s.BOOL_IDX],
                                 [mosek.boundkey.ra] * num_bool,
                                 [0] * num_bool, [1] * num_bool)
            task.putvartypelist(data[s.INT_IDX], [mosek.variabletype.type_int] * num_int)

        # Define linear inequality and equality constraints.
        #
        #   Mosek will see a total of m linear expressions, which must
        #   define linear inequalities and equalities. The variable x
        #   contributes to these linear expressions by standard
        #   matrix-vector multiplication; the matrix in question is
        #   referred to as "A" in the mosek documentation. The PSD
        #   variables have a different means of contributing to the
        #   linear expressions. Specifically, a PSD variable Xj contributes
        #   "+tr( \bar{A}_{ij} * Xj )" to the i-th linear expression,
        #   where \bar{A}_{ij} is specified by a call to putbaraij.
        #
        #   The following code has three phases.
        #       (1) Build the matrix A.
        #       (2) Specify the \bar{A}_{ij} for PSD variables.
        #       (3) Specify the RHS of the m linear (in)equalities.
        #
        #   Remark : The parameter G gives every row in the first
        #   n0 columns of A. The remaining columns of A are for SOC
        #   and EXP slack variables. We can actually account for all
        #   of these slack variables at once by specifying a giant
        #   identity matrix in the appropriate position in A.

        task.appendcons(m)
        row, col, vals = sp.sparse.find(G)
        task.putaijlist(row.tolist(), col.tolist(), vals.tolist())
        total_soc_exp_slacks = sum(dims[s.SOC_DIM]) + 3 * dims[s.EXP_DIM]
        if total_soc_exp_slacks > 0:
            i = dims[s.LEQ_DIM] + dims[s.EQ_DIM]  # constraint index in {0, ..., m - 1}
            j = len(c)  # index of the first slack variable in the block vector "x".
            rows = np.arange(i, i + total_soc_exp_slacks).tolist()
            cols = np.arange(j, j + total_soc_exp_slacks).tolist()
            task.putaijlist(rows, cols, [1] * total_soc_exp_slacks)

        # constraint index; start of LMIs.
        i = dims[s.LEQ_DIM] + dims[s.EQ_DIM] + total_soc_exp_slacks
        for j, dim in enumerate(dims[s.PSD_DIM]):  # SDP slack variable "Xj"
            for row_idx in range(dim):
                for col_idx in range(dim):
                    val = 1. if row_idx == col_idx else 0.5
                    row = max(row_idx, col_idx)
                    col = min(row_idx, col_idx)
                    mat = task.appendsparsesymmat(dim, [row], [col], [val])
                    task.putbaraij(i, j, [mat], [1.0])
                    i += 1

        num_eq = len(h) - dims[s.LEQ_DIM]
        type_constraint = [mosek.boundkey.up] * dims[s.LEQ_DIM] + \
                          [mosek.boundkey.fx] * num_eq
        task.putconboundlist(np.arange(m, dtype=int), type_constraint, h, h)

        # Define the objective, and optimize the mosek task.

        task.putclist(np.arange(len(c)), c)
        task.putobjsense(mosek.objsense.minimize)
        if save_file:
            task.writedata(save_file)
        task.optimize()

        if verbose:
            task.solutionsummary(mosek.streamtype.msg)

        return {'env': env, 'task': task, 'solver_options': solver_opts}

    def old_invert(self, results, inverse_data):
        """
        Use information contained within "results" and "inverse_data" to properly
        define a cvxpy Solution object.

        :param results: a dictionary with three key-value pairs:
            results['env'] == the mosek Environment object generated during solve_via_data,
            results['task'] == the mosek Task object generated during solve_via_data,
            results['solver_options'] == the dictionary of parameters passed to solve_via_data.
        :param inverse_data: data recorded during "apply".

        :return: a cvxpy Solution object, instantiated with the following fields:

            (1) status - the closest cvxpy analog of mosek's status code.
            (2) opt_val - the optimal objective function value
            (after translation by a possible constant).
            (3) primal_vars - a dictionary with a single element: "z", represented as a list.
            (4) dual_vars - a dictionary with as many elements as
            constraints in the cvxpy standard form problem.
                The elements of the dictionary are either scalars, or numpy arrays.
        """
        import mosek
        # Status map is taken from:
        # https://docs.mosek.com/8.1/pythonapi/constants.html?highlight=solsta#mosek.solsta
        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.integer_optimal: s.OPTIMAL,
                      mosek.solsta.prim_feas: s.OPTIMAL_INACCURATE,    # for integer problems
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED}
        # "Near" statuses only up to Mosek 8.1
        if hasattr(mosek.solsta, 'near_optimal'):
            STATUS_MAP_INACCURATE = {mosek.solsta.near_optimal: s.OPTIMAL_INACCURATE,
                                     mosek.solsta.near_integer_optimal: s.OPTIMAL_INACCURATE,
                                     mosek.solsta.near_prim_infeas_cer: s.INFEASIBLE_INACCURATE,
                                     mosek.solsta.near_dual_infeas_cer: s.UNBOUNDED_INACCURATE}
            STATUS_MAP.update(STATUS_MAP_INACCURATE)
        STATUS_MAP = defaultdict(lambda: s.SOLVER_ERROR, STATUS_MAP)

        env = results['env']
        task = results['task']
        solver_opts = results['solver_options']

        if inverse_data['integer_variables']:
            sol = mosek.soltype.itg
        elif 'bfs' in solver_opts and solver_opts['bfs'] and inverse_data['is_LP']:
            sol = mosek.soltype.bas  # the basic feasible solution
        else:
            sol = mosek.soltype.itr  # the solution found via interior point method

        problem_status = task.getprosta(sol)
        solution_status = task.getsolsta(sol)

        status = STATUS_MAP[solution_status]

        # For integer problems, problem status determines infeasibility (no solution)
        if sol == mosek.soltype.itg and problem_status == mosek.prosta.prim_infeas:
            status = s.INFEASIBLE

        if status in s.SOLUTION_PRESENT:
            # get objective value
            opt_val = task.getprimalobj(sol) + inverse_data[s.OBJ_OFFSET]
            # recover the cvxpy standard-form primal variable
            z = [0.] * inverse_data['n0']
            task.getxxslice(sol, 0, len(z), z)
            primal_vars = {inverse_data[self.VAR_ID]: z}
            # recover the cvxpy standard-form dual variables
            if sol == mosek.soltype.itg:
                dual_vars = None
            else:
                dual_vars = MOSEK.recover_dual_variables(task, sol, inverse_data)
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None
            primal_vars = None
            dual_vars = None

        # Store computation time
        attr = {s.SOLVE_TIME: task.getdouinf(mosek.dinfitem.optimizer_time)}

        # Delete the mosek Task and Environment
        task.__exit__(None, None, None)
        env.__exit__(None, None, None)

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def apply(self, problem):
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        if problem.x.boolean_idx or problem.x.integer_idx:  # check if either list is empty
            data, inv_data = Slacks.apply(problem, [a2d.NONNEG])
        else:
            data, inv_data = Dualize.apply(problem)
        return data, inv_data

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import mosek
        env = mosek.Env()
        task = env.Task(0, 0)
        save_file = self.handle_options(env, task, verbose, solver_opts)

        ## Check if the cvxpy standard form has zero variables. If so,
        ## return a trivial solution. This is necessary because MOSEK
        ## will crash if handed a problem with zero variables.
        #if len(data[s.C]) == 0:
        #    return {s.STATUS: s.OPTIMAL, s.PRIMAL: [],
        #            s.VALUE: data[s.OFFSET], s.EQ_DUAL: [], s.INEQ_DUAL: []}

        if 'dualized' in data:
            task = self._build_dualized_task(task, data)
        else:
            task = self._build_slack_task(task, data)

        # Optimize the Mosek Task and return the result.
        if save_file:
            task.writedata(save_file)
        task.optimize()

        if verbose:
            task.solutionsummary(mosek.streamtype.msg)

        return {'env': env, 'task': task, 'solver_options': solver_opts}

    def _build_dualized_task(self, task, data):
        # problem has A, b, K_dir, c.
        #   max{ c.T @ x : A @ x == b, x in K_dir }
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
            task.appendconesseq([mosek.conetype.quad] * num_soc, [0] * num_soc,
                                K[a2d.SOC], idx)
            idx += sum(K[a2d.SOC])
        num_psd = len(K[a2d.PSD])
        if num_psd > 0:
            raise NotImplementedError()
        num_dexp = K[a2d.DUAL_EXP]
        if num_dexp > 0:
            task.appendconesseq([mosek.conetype.dexp] * num_dexp, [0] * num_dexp,
                                [3] * num_dexp, idx)
            idx += 3 * num_dexp
        return task

    def _build_slack_task(self, task, data):
        """
        min{ c.T @ x : A @ x <=_{K_aff} b,  x in K_dir }
        """
        import mosek
        K_aff = data['K_aff']
        # K_aff keyed by a2d.NONNEG, a2d.ZERO
        c, A, b = data[s.C], data[s.A], data[s.B]
        # The rows of (A, b) go by a2d.FREE then a2d.NONNEG
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

        if task.getnumintvar() > 0:
            sol_type = mosek.soltype.itg
        elif 'bfs' in solver_opts and solver_opts['bfs'] and task.getnumcone() == 0:
            sol_type = mosek.soltype.bas  # the basic feasible solution
        else:
            sol_type = mosek.soltype.itr  # the solution found via interior point method

        problem_status = task.getprosta(sol_type)
        if sol_type == mosek.soltype.itg and problem_status == mosek.prosta.prim_infeas:
            status = s.INFEASIBLE
        else:
            solsta = task.getsolsta(sol_type)
            status = STATUS_MAP[solsta]

        prob_val = np.NaN
        prim_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            prob_val = task.getprimalobj(sol_type)
            K = inverse_data['K_dir']
            prim_vars = self.recover_primal_variables(task, sol_type, K)
            dual_vars = self.recover_dual_variables(task, sol_type)
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

    def recover_dual_variables(self, task, sol):
        # This function is only designed to recover dual variables
        # when the "dualize" transformation has been applied.
        # A problem is dualized if and only if it has no discrete variables.
        if task.getnumintvar() == 0:
            dual_vars = dict()
            dual_var = [0.] * task.getnumcon()
            task.gety(sol, dual_var)
            dual_vars[s.EQ_DUAL] = np.array(dual_var)
        else:
            dual_vars = None
        return dual_vars

    def recover_primal_variables(self, task, sol, K_dir):
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
        num_psd = len(K_dir[a2d.PSD])
        if num_psd > 0:
            raise NotImplementedError()
        num_dexp = K_dir[a2d.DUAL_EXP]
        if num_dexp > 0:
            temp = [0.] * (3 * num_dexp)
            task.getxxslice(sol, idx, idx + len(temp), temp)
            temp = np.array(temp)
            perm = expcone_permutor(num_dexp, MOSEK.EXP_CONE_ORDER)
            prim_vars[a2d.DUAL_EXP] = temp[perm]
            idx += (3 * num_dexp)
        return prim_vars

    @staticmethod
    def old_recover_dual_variables(task, sol, inverse_data):
        """
        A cvxpy Constraint "constr" views itself as
            affine_expression(z) in K.
        The "apply(...)" function represents constr as
            G * z <=_K h
        for appropriate arrays (G, h).
        After adding slack variables, constr becomes
            G * z + s == h, s in K.
        From "apply(...)" and "solve_via_data(...)", one will find
            affine_expression(z) == h - G * z == s.
        As a result, the dual variable suitable for "constr" is
        the conic dual variable to the constraint "s in K".

        Mosek documentation refers to conic dual variables as follows:
            zero cone: 'y'
            nonnegative orthant: 'suc'
            second order cone: 'snx'
            exponential cone: 'snx'
            PSD cone: 'barsj'.

        :param task: the mosek task object which was just optimized
        :param sol: the mosek solution type (usually mosek.soltype.itr,
        but possibly mosek.soltype.bas if the problem was a linear
        program and the user requested a basic feasible solution).
        :param inverse_data: data recorded during "apply(...)".

        :return: a dictionary mapping a cvxpy constraint object's id to its
        corresponding dual variables in the current solution.
        """
        dual_vars = dict()

        # Dual variables for the inequality constraints
        suc_len = sum(ell for _, ell in inverse_data['suc_slacks'])
        if suc_len > 0:
            suc = [0.] * suc_len
            task.getsucslice(sol, 0, suc_len, suc)
            dual_vars.update(MOSEK._parse_dual_var_block(suc, inverse_data['suc_slacks']))

        # Dual variables for the original equality constraints
        y_len = sum(ell for _, ell in inverse_data['y_slacks'])
        if y_len > 0:
            y = [0.] * y_len
            task.getyslice(sol, suc_len, suc_len + y_len, y)
            y = [-val for val in y]
            dual_vars.update(MOSEK._parse_dual_var_block(y, inverse_data['y_slacks']))

        # Dual variables for SOC and EXP constraints
        snx_len = sum(ell for _, ell in inverse_data['snx_slacks'])
        if snx_len > 0:
            snx = np.zeros(snx_len)
            task.getsnxslice(sol, inverse_data['n0'], inverse_data['n0'] + snx_len, snx)
            dual_vars.update(MOSEK._parse_dual_var_block(snx, inverse_data['snx_slacks']))

        # Dual variables for PSD constraints
        for j, (id, dim) in enumerate(inverse_data['psd_dims']):
            sj = [0.] * (dim * (dim + 1) // 2)
            task.getbarsj(sol, j, sj)
            dual_vars[id] = vectorized_lower_tri_to_mat(sj, dim)

        # Now that all dual variables have been recovered, find those corresponding
        # to the exponential cone, and permute the entries to reflect the CVXPY
        # standard for the exponential cone.
        for con in inverse_data['constraints']:
            if isinstance(con, ExpCone):
                cid = con.id
                perm = expcone_permutor(con.num_cones(), MOSEK.EXP_CONE_ORDER)
                dual_vars[cid] = dual_vars[cid][perm]
        return dual_vars

    @staticmethod
    def _old_parse_dual_var_block(dual_var, constr_id_to_constr_dim):
        """
        :param dual_var: a list of numbers returned by some 'get dual variable'
          function in mosek's Optimzer API.
        :param constr_id_to_constr_dim: a list of tuples (id, dim).
          The entry "id" is the index of the cvxpy Constraint
          object to which the next "dim" entries of the dual variable belong.

        :return: a dictionary keyed by cvxpy Constraint object indicies,
          with either scalar or numpy array values.
        """
        dual_vars = dict()
        running_idx = 0
        for id, dim in constr_id_to_constr_dim:
            if dim == 1:
                dual_vars[id] = dual_var[running_idx]  # a scalar
            else:
                dual_vars[id] = np.array(dual_var[running_idx:(running_idx + dim)])
            running_idx += dim
        return dual_vars

    def handle_options(self, env, task, verbose, solver_opts):
        # If verbose, then set default logging parameters.
        import mosek

        if verbose:
            import sys

            def streamprinter(text):
                sys.stdout.write(text)
                sys.stdout.flush()

            print('\n')
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)
            task.putintparam(mosek.iparam.infeas_report_auto, mosek.onoffkey.on)
            task.putintparam(mosek.iparam.log_presolve, 0)

        # Parse all user-specified parameters (override default logging
        # parameters if applicable).
        kwargs = sorted(solver_opts.keys())
        save_file = None
        bfs = False
        if 'mosek_params' in kwargs:
            self._handle_mosek_params(task, solver_opts['mosek_params'])
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

    @staticmethod
    def _handle_mosek_params(task, params):
        if params is None:
            return

        import mosek

        def _handle_str_param(param, value):
            if param.startswith("MSK_DPAR_"):
                task.putnadouparam(param, value)
            elif param.startswith("MSK_IPAR_"):
                task.putnaintparam(param, value)
            elif param.startswith("MSK_SPAR_"):
                task.putnastrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        def _handle_enum_param(param, value):
            if isinstance(param, mosek.dparam):
                task.putdouparam(param, value)
            elif isinstance(param, mosek.iparam):
                task.putintparam(param, value)
            elif isinstance(param, mosek.sparam):
                task.putstrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        for param, value in params.items():
            if isinstance(param, str):
                _handle_str_param(param.strip(), value)
            else:
                _handle_enum_param(param, value)
