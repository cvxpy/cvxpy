"""
This file is the CVXPY conic extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, SvecPSD
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT
from cvxpy.utilities.psd_utils import TriangleKind


def _add_psd_bound_rows(A, b, dims, lb, ub):
    """Add finite variable bounds as explicit inequality rows.

    The rows are inserted right after the existing NonNeg (LEQ) block so the
    cone partition passed to ``loadConeMatrix`` stays contiguous: COPT expects
    rows ordered as free, linear, SOC, exponential, PSD.
    """
    n = A.shape[1]
    extra_rows = []
    extra_b = []

    if ub is not None:
        finite_ub = np.isfinite(ub)
        n_ub = int(np.count_nonzero(finite_ub))
        if n_ub:
            extra_rows.append(_bound_selector(n_ub, n, np.flatnonzero(finite_ub), 1.0))
            extra_b.append(ub[finite_ub])

    if lb is not None:
        finite_lb = np.isfinite(lb)
        n_lb = int(np.count_nonzero(finite_lb))
        if n_lb:
            extra_rows.append(_bound_selector(n_lb, n, np.flatnonzero(finite_lb), -1.0))
            extra_b.append(-lb[finite_lb])

    if extra_rows:
        insert_at = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        bound_block = sp.vstack(extra_rows, format='csr')
        A = sp.vstack([A[:insert_at], bound_block, A[insert_at:]], format='csc')
        b = np.concatenate([b[:insert_at], *extra_b, b[insert_at:]])
        dims[s.LEQ_DIM] += bound_block.shape[0]

    return A, b


def _bound_selector(num_rows, num_cols, cols, value):
    return sp.csr_array((np.full(num_rows, value), (np.arange(num_rows), cols)),
                        shape=(num_rows, num_cols))


class COPT(ConicSolver):
    """
    An interface for the COPT solver.
    """
    # Solver capabilities
    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, SvecPSD]
    REQUIRES_CONSTR = True
    PSD_TRIANGLE_KIND = TriangleKind.LOWER
    PSD_SQRT2_SCALING = False

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
                  4: s.INFEASIBLE_OR_UNBOUNDED,  # infeasible or unbounded
                  5: s.SOLVER_ERROR,        # numerical
                  6: s.USER_LIMIT,          # node limit
                  7: s.OPTIMAL_INACCURATE,  # imprecise
                  8: s.USER_LIMIT,          # time out
                  9: s.SOLVER_ERROR,        # unfinished
                  10: s.USER_LIMIT,         # interrupted
                  11: s.USER_LIMIT,         # iteration limit
                  20: s.OPTIMAL,            # local optimal
                  21: s.INFEASIBLE          # local infeasible
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

    def _dual_vars(self, solution, inverse_data):
        """Map the stacked ``[eq; ineq]`` dual vector to a CVXPY dual dict.

        Shared by the optimal duals and the infeasibility (Farkas) certificate,
        which use the same row layout.
        """
        eq_dual = utilities.get_dual_values(
            solution[s.EQ_DUAL],
            utilities.extract_dual_value,
            inverse_data[COPT.EQ_CONSTR])
        leq_dual = utilities.get_dual_values(
            solution[s.INEQ_DUAL],
            utilities.extract_dual_value,
            inverse_data[COPT.NEQ_CONSTR])
        for con in inverse_data[self.NEQ_CONSTR]:
            if isinstance(con, ExpCone):
                cid = con.id
                n_cones = con.num_cones()
                perm = utilities.expcone_permutor(n_cones, COPT.EXP_CONE_ORDER)
                leq_dual[cid] = leq_dual[cid][perm]
        eq_dual.update(leq_dual)
        return eq_dual

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]
        attr = {s.SOLVE_TIME: solution[s.SOLVE_TIME],
                s.NUM_ITERS: solution[s.NUM_ITERS],
                s.EXTRA_STATS: solution['model']}

        # EQ_DUAL/INEQ_DUAL hold the constraint duals when a solution is present
        # and the dual Farkas infeasibility certificate otherwise.
        dual_vars = None
        if s.EQ_DUAL in solution and not inverse_data['is_mip']:
            dual_vars = self._dual_vars(solution, inverse_data)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution[s.PRIMAL]}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr, dual_vars)

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

            # For PSD path (dualized), variable bounds must be added as
            # explicit constraints since loadConeMatrix doesn't support bounds.
            psd_lb = data[s.LOWER_BOUNDS]
            psd_ub = data[s.UPPER_BOUNDS]
            if psd_lb is not None or psd_ub is not None:
                A, b = _add_psd_bound_rows(A, b, dims, psd_lb, psd_ub)

            # Solve the dualized problem
            rowmap = model.loadConeMatrix(-b, A.transpose().tocsc(), -c, dims)
            model.objsense = copt.COPT.MAXIMIZE
        else:
            # Build problem data
            n = data[s.C].shape[0]

            c = data[s.C]
            A = data[s.A]

            lhs = np.copy(data[s.B])
            lhs[range(dims[s.EQ_DIM], dims[s.EQ_DIM] + dims[s.LEQ_DIM])] = -copt.COPT.INFINITY
            rhs = np.copy(data[s.B])

            lb = data[s.LOWER_BOUNDS]
            ub = data[s.UPPER_BOUNDS]
            if lb is None:
                lb = np.full(n, -copt.COPT.INFINITY)
            else:
                lb = np.copy(lb)
            if ub is None:
                ub = np.full(n, +copt.COPT.INFINITY)
            else:
                ub = np.copy(ub)

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
            model.loadMatrix(c, A.tocsc(), lhs, rhs, lb, ub, vtype)

            # Load cone data
            if dims[s.SOC_DIM]:
                model.loadCone(ncone, None, dims[s.SOC_DIM],
                               range(A.shape[1] - nconedim - nexpconedim, A.shape[1] - nexpconedim))

            # Load exponential cone data
            if dims[s.EXP_DIM]:
                model.loadExpCone(nexpcone, None,
                                  range(A.shape[1] - nexpconedim, A.shape[1]))

        # Request a dual Farkas ray so an infeasibility certificate is available
        # for infeasible problems (not the dualized PSD path, and not MIPs). Set
        # it before the user-settings loop so an explicit value in solver_opts wins.
        if not dims[s.PSD_DIM] and not (data[s.BOOL_IDX] or data[s.INT_IDX]):
            model.setParam(copt.COPT.Param.ReqFarkasRay, 1)

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

        # On infeasibility, return the dual Farkas ray as the certificate,
        # negated to match CVXPY's sign convention (as done for optimal duals).
        # Split [eq; ineq] at EQ_DIM, mirroring the optimal-dual extraction.
        if (not dims[s.PSD_DIM]
                and solution[s.STATUS] in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE)
                and not (data[s.BOOL_IDX] or data[s.INT_IDX])
                and model.hasdualfarkas):
            y = -np.array(model.getInfo(copt.COPT.Info.DualFarkas, model.getConstrs()))
            solution[s.EQ_DUAL] = y[0:dims[s.EQ_DIM]]
            solution[s.INEQ_DUAL] = y[dims[s.EQ_DIM]:]

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
