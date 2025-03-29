"""
This file is the CVXPY conic extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT


def tri_to_full(lower_tri, n):
    """
    Expands n*(n+1)//2 lower triangular to full matrix

    Parameters
    ----------
    lower_tri : numpy.ndarray
        A NumPy array representing the lower triangular part of the
        matrix, stacked in column-major order.
    n : int
        The number of rows (columns) in the full square matrix.

    Returns
    -------
    numpy.ndarray
        A 2-dimensional ndarray that is the scaled expansion of the lower
        triangular array.
    """
    full = np.zeros((n, n))
    full[np.triu_indices(n)] = lower_tri
    full += full.T
    full[np.diag_indices(n)] /= 2.0
    return np.reshape(full, n*n, order="F")


class COPT(ConicSolver):
    """
    An interface for the COPT solver.
    """
    # Solver capabilities
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PSD]
    REQUIRES_CONSTR = True

    EXP_CONE_ORDER = [2, 1, 0]

    # Support MILP and MISOCP
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map between COPT status and CVXPY status
    STATUS_MAP = {
                  1: s.OPTIMAL,             # optimal
                  2: s.INFEASIBLE,          # infeasible
                  3: s.UNBOUNDED,           # unbounded
                  4: s.INF_OR_UNB,          # infeasible or unbounded
                  5: s.SOLVER_ERROR,        # numerical
                  6: s.USER_LIMIT,          # node limit
                  7: s.OPTIMAL_INACCURATE,  # imprecise
                  8: s.USER_LIMIT,          # time out
                  9: s.SOLVER_ERROR,        # unfinished
                  10: s.USER_LIMIT          # interrupted
                 }

    def name(self):
        """
        The name of solver.
        """
        return 'COPT'

    def import_solver(self):
        """
        Imports the solver.
        """
        import coptpy  # noqa F401

    def accepts(self, problem):
        """
        Can COPT solve the problem?
        """
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    @staticmethod
    def psd_format_mat(constr):
        """
        Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as COPT expects constraints to be
        imposed on solely the lower triangular part of the variable matrix.
        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1)//2

        row_arr = np.arange(0, entries)

        lower_diag_indices = np.tril_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(lower_diag_indices,
                                               (rows, cols),
                                               order='F'))

        val_arr = np.zeros((rows, cols))
        val_arr[lower_diag_indices] = 1.0
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]

        shape = (entries, rows*cols)
        scaled_lower_tri = sp.csc_array((val_arr, (row_arr, col_arr)), shape)

        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_array((val_symm, (row_symm, col_symm)))

        return scaled_lower_tri @ symm_matrix

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """
        Extracts the dual value for constraint starting at offset.

        Special cases PSD constraints, as per the COPT specification.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            lower_tri_dim = dim * (dim + 1) // 2
            new_offset = offset + lower_tri_dim
            lower_tri = result_vec[offset:new_offset]
            full = tri_to_full(lower_tri, dim)
            return full, new_offset
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def apply(self, problem):
        """
        Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        data, inv_data = super(COPT, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]
        attr = {s.SOLVE_TIME: solution[s.SOLVE_TIME],
                s.NUM_ITERS: solution[s.NUM_ITERS],
                s.EXTRA_STATS: solution['model']}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution[s.PRIMAL]}
            if not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution[s.EQ_DUAL],
                    self.extract_dual_value,
                    inverse_data[COPT.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution[s.INEQ_DUAL],
                    self.extract_dual_value,
                    inverse_data[COPT.NEQ_CONSTR])
                for con in inverse_data[self.NEQ_CONSTR]:
                    if isinstance(con, ExpCone):
                        cid = con.id
                        n_cones = con.num_cones()
                        perm = utilities.expcone_permutor(n_cones, COPT.EXP_CONE_ORDER)
                        leq_dual[cid] = leq_dual[cid][perm]
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

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
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import coptpy as copt

        # Create COPT environment and model
        envconfig = copt.EnvrConfig()
        if not verbose:
            envconfig.set('nobanner', '1')

        env = copt.Envr(envconfig)
        model = env.createModel()

        # Pass through verbosity
        model.setParam(copt.COPT.Param.Logging, verbose)

        # Get the dimension data
        dims = dims_to_solver_dict(data[s.DIMS])

        # Treat cone problem with PSD part specially
        rowmap = None
        if dims[s.PSD_DIM]:
            # Build cone problem data
            c = data[s.C]
            A = data[s.A]
            b = data[s.B]

            # Solve the dualized problem
            # TODO switch to `A.transpose().tocsc()` when COPT supports sparray
            rowmap = model.loadConeMatrix(-b, sp.csc_matrix(A.transpose()), -c, dims)
            model.objsense = copt.COPT.MAXIMIZE
        else:
            # Build problem data
            n = data[s.C].shape[0]

            c = data[s.C]
            A = data[s.A]

            lhs = np.copy(data[s.B])
            lhs[range(dims[s.EQ_DIM], dims[s.EQ_DIM] + dims[s.LEQ_DIM])] = -copt.COPT.INFINITY
            rhs = np.copy(data[s.B])

            lb = np.full(n, -copt.COPT.INFINITY)
            ub = np.full(n, +copt.COPT.INFINITY)

            vtype = None
            if data[s.BOOL_IDX] or data[s.INT_IDX]:
                vtype = np.array([copt.COPT.CONTINUOUS] * n)
                if data[s.BOOL_IDX]:
                    vtype[data[s.BOOL_IDX]] = copt.COPT.BINARY
                    lb[data[s.BOOL_IDX]] = 0
                    ub[data[s.BOOL_IDX]] = 1
                if data[s.INT_IDX]:
                    vtype[data[s.INT_IDX]] = copt.COPT.INTEGER

            # Build cone data
            ncone = 0
            nconedim = 0
            if dims[s.SOC_DIM]:
                ncone = len(dims[s.SOC_DIM])
                nconedim = sum(dims[s.SOC_DIM])
                nlinrow = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
                nlincol = A.shape[1]

                diag = sp.diags_array(np.ones(nconedim), offsets=-nlinrow,
                                shape=(A.shape[0], nconedim))
                A = sp.hstack([A, diag], format='csc')

                c = np.append(c, np.zeros(nconedim))

                lb = np.append(lb, -copt.COPT.INFINITY * np.ones(nconedim))
                ub = np.append(ub, +copt.COPT.INFINITY * np.ones(nconedim))

                lb[nlincol] = 0.0
                if len(dims[s.SOC_DIM]) > 1:
                    for dim in dims[s.SOC_DIM][:-1]:
                        nlincol += dim
                        lb[nlincol] = 0.0

                if vtype is not None:
                    vtype = np.append(vtype, [copt.COPT.CONTINUOUS] * nconedim)

            # Build exponential cone data
            nexpcone = 0
            nexpconedim = 0
            if dims[s.EXP_DIM]:
                nexpcone = dims[s.EXP_DIM]
                nexpconedim = dims[s.EXP_DIM] * 3
                nlinrow = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
                if dims[s.SOC_DIM]:
                    nlinrow += sum(dims[s.SOC_DIM])
                nlincol = A.shape[1]

                diag = sp.diags_array(np.ones(nexpconedim), offsets=-nlinrow,
                                      shape=(A.shape[0], nexpconedim))
                A = sp.hstack([A, diag], format='csc')

                c = np.append(c, np.zeros(nexpconedim))

                lb = np.append(lb, -copt.COPT.INFINITY * np.ones(nexpconedim))
                ub = np.append(ub, +copt.COPT.INFINITY * np.ones(nexpconedim))

                if vtype is not None:
                    vtype = np.append(vtype, [copt.COPT.CONTINUOUS] * nexpconedim)

            # Load matrix data
            # TODO remove `sp.csc_matrix` when COPT starts supporting sparray
            model.loadMatrix(c, sp.csc_matrix(A), lhs, rhs, lb, ub, vtype)

            # Load cone data
            if dims[s.SOC_DIM]:
                model.loadCone(ncone, None, dims[s.SOC_DIM],
                               range(A.shape[1] - nconedim - nexpconedim, A.shape[1] - nexpconedim))

            # Load exponential cone data
            if dims[s.EXP_DIM]:
                model.loadExpCone(nexpcone, None,
                                  range(A.shape[1] - nexpconedim, A.shape[1]))

        # Set parameters
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        solution = {}
        try:
            model.solve()
            # Reoptimize if INF_OR_UNBD, to get definitive answer.
            if model.status == copt.COPT.INF_OR_UNB and solver_opts.get('reoptimize', True):
                model.setParam(copt.COPT.Param.Presolve, 0)
                model.solve()

            if dims[s.PSD_DIM]:
                if model.haslpsol:
                    solution[s.VALUE] = model.objval

                    # Recover the primal solution
                    nrow = len(c)
                    duals = model.getDuals()
                    psdduals = model.getPsdDuals()
                    y = np.zeros(nrow)
                    for i in range(nrow):
                        if rowmap[i] < 0:
                            y[i] = -psdduals[-rowmap[i] - 1]
                        else:
                            y[i] = -duals[rowmap[i] - 1]
                    solution[s.PRIMAL] = y

                    # Recover the dual solution
                    solution['y'] = np.hstack((model.getValues(), model.getPsdValues()))
                    solution[s.EQ_DUAL] = solution['y'][0:dims[s.EQ_DIM]]
                    solution[s.INEQ_DUAL] = solution['y'][dims[s.EQ_DIM]:]
            else:
                if model.haslpsol or model.hasmipsol:
                    solution[s.VALUE] = model.objval
                    solution[s.PRIMAL] = np.array(model.getValues())

                # Get dual values of linear constraints if not MIP
                if not (data[s.BOOL_IDX] or data[s.INT_IDX]) and model.haslpsol:
                    solution['y'] = -np.array(model.getDuals())
                    solution[s.EQ_DUAL] = solution['y'][0:dims[s.EQ_DIM]]
                    solution[s.INEQ_DUAL] = solution['y'][dims[s.EQ_DIM]:]
        except Exception:
            pass

        solution[s.SOLVE_TIME] = model.solvingtime
        solution[s.NUM_ITERS] = model.barrieriter + model.simplexiter

        if dims[s.PSD_DIM]:
            if model.status == copt.COPT.INFEASIBLE:
                solution[s.STATUS] = s.UNBOUNDED
            elif model.status == copt.COPT.UNBOUNDED:
                solution[s.STATUS] = s.INFEASIBLE
            else:
                solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        else:
            solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        if solution[s.STATUS] == s.USER_LIMIT and model.hasmipsol:
            solution[s.STATUS] = s.OPTIMAL_INACCURATE
        if solution[s.STATUS] == s.USER_LIMIT and not model.hasmipsol:
            solution[s.STATUS] = s.INFEASIBLE_INACCURATE

        solution['model'] = model

        return solution
    
    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COPT"]