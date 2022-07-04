"""
This file is the CVXPY extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver, dims_to_solver_dict,)


class COPT(ConicSolver):
    """
    An interface for the COPT solver.
    """

    # Solver capabilities
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    # Only supports MI LPs.
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    # Map between COPT status and CVXPY status
    STATUS_MAP = {
                  1: s.OPTIMAL,       # optimal
                  2: s.INFEASIBLE,    # infeasible
                  3: s.UNBOUNDED,     # unbounded
                  4: s.INF_OR_UNB,    # infeasible or unbounded
                  5: s.SOLVER_ERROR,  # numerical
                  6: s.USER_LIMIT,    # node limit
                  7: s.SOLVER_ERROR,  # error
                  8: s.USER_LIMIT,    # time out
                  9: s.SOLVER_ERROR,  # unfinished
                  10: s.USER_LIMIT    # interrupted
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
        import coptpy
        coptpy  # For flake8

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
        status = solution['status']
        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME]}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution['primal']}
            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[COPT.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[COPT.NEQ_CONSTR])
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
        env = copt.Envr()
        model = env.createModel()

        # Pass through verbosity
        model.setParam(copt.COPT.Param.Logging, verbose)

        # Get the dimension data
        dims = dims_to_solver_dict(data[s.DIMS])

        # Build problem data
        n = data[s.C].shape[0]

        c = data[s.C]
        A = sp.csc_matrix(data[s.A])

        lhs = np.copy(data[s.B])
        lhs[range(dims[s.EQ_DIM], dims[s.EQ_DIM] + dims[s.LEQ_DIM])] = -copt.COPT.INFINITY
        rhs = np.copy(data[s.B])

        lb = np.empty(n)
        lb.fill(-copt.COPT.INFINITY)
        ub = np.empty(n)
        ub.fill(+copt.COPT.INFINITY)

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

            diag = sp.spdiags(np.ones(nconedim), -nlinrow, A.shape[0], nconedim)
            A = sp.csc_matrix(sp.hstack([A, diag]))

            c = np.append(c, np.zeros(nconedim))

            lb = np.append(lb, -copt.COPT.INFINITY * np.ones(nconedim))
            ub = np.append(ub, +copt.COPT.INFINITY * np.ones(nconedim))

            lb[nlincol] = 0.0
            if len(dims[s.SOC_DIM]) > 1:
                for dim in dims[s.SOC_DIM][:-1]:
                    nlincol += dim
                    lb[nlincol] = 0.0

            if data[s.BOOL_IDX] or data[s.INT_IDX]:
                vtype = np.append(vtype, [copt.COPT.CONTINUOUS] * nconedim)

        # Load matrix data
        model.loadMatrix(c, A, lhs, rhs, lb, ub, vtype)

        # Load cone data
        if dims[s.SOC_DIM]:
            model.loadCone(ncone, None, dims[s.SOC_DIM], range(A.shape[1] - nconedim, A.shape[1]))

        # Set parameters
        for key, value in solver_opts.items():
            model.setParam(key, value)

        solution = {}
        try:
            model.solve()
            # Reoptimize if INF_OR_UNBD, to get definitive answer.
            if model.status == copt.COPT.INF_OR_UNB and solver_opts.get('reoptimize', True):
                model.setParam(copt.COPT.Param.Presolve, 0)
                model.solve()
            solution["value"] = model.objval
            solution["primal"] = np.array(model.getValues())

            # Get dual values of linear constraints if not MIP
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                solution["y"] = -np.array(model.getDuals())
                solution[s.EQ_DUAL] = solution["y"][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = solution["y"][dims[s.EQ_DIM]:]
        except Exception:
            pass

        solution[s.SOLVE_TIME] = model.solvingtime

        solution["status"] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        if solution["status"] == s.USER_LIMIT and model.hasmipsol:
            solution["status"] = s.OPTIMAL_INACCURATE
        if solution["status"] == s.USER_LIMIT and not model.hasmipsol:
            solution["status"] = s.INFEASIBLE_INACCURATE

        solution["model"] = model

        return solution
