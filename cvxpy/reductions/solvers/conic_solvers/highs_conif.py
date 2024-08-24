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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)


class HIGHS(ConicSolver):
    """
    An interface for the HiGHS solver.
    """

    #solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    def __init__(self) -> None:
        self.prob_ = None
    
    def name(self):
        """The name of the solver
        """
        return s.HIGHS
    
    def import_solver(self) -> None:
        """Imports the solver.
        """
        import highspy
        self.version = highspy.Highs.version

    def accepts(self, problem) -> bool:
        """can HiGHS solve the problem?
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
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(HIGHS, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']
        attr = {s.EXTRA_STATS: solution['model'],
                s.SOLVE_TIME: solution[s.SOLVE_TIME]}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[HIGHS.VAR_ID]: solution['primal']}
            if "eq_dual" in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[HIGHS.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[HIGHS.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

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
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import highspy as hp

        dims = dims_to_solver_dict(data[s.DIMS])
        nrowsEQ = dims[s.EQ_DIM]
        nrowsLEQ = dims[s.LEQ_DIM]
        nrows = nrowsEQ + nrowsLEQ

        c = data[s.C]
        #linear constraints Ax <= b
        b = data[s.B][:nrows]
        A = sp.csr_matrix(data[s.A][:nrows])
        
        nvars = len(c)
        
        self.prob_ = hp.Highs()
        inf = self.prob_.inf

        if verbose:
            self.prob_.setOptionValue("log_to_console", True)
        else:
            self.prob_.setOptionValue("log_to_console", False)

        self.prob_.addVars(nvars, [-inf]*nvars, [inf]*nvars)
        if data[s.BOOL_IDX] != []:
            nbinaryvars = len(data[s.BOOL_IDX])
            self.prob_.changeColsBounds(
                nbinaryvars,
                data[s.BOOL_IDX],
                [0.0] * nbinaryvars,
                [1.0] * nbinaryvars
            )
            self.prob_.changeColsIntegrality(
                nbinaryvars,
                data[s.BOOL_IDX],
                [hp.HighsVarType.kInteger] * nbinaryvars
            )

        if data[s.INT_IDX] != []:
            nintvars = len(data[s.INT_IDX])
            self.prob_.changeColsIntegrality(
                nintvars,
                data[s.INT_IDX],
                [hp.HighsVarType.kInteger] * nintvars
            )
        self.prob_.changeColsCost(nvars, np.array(range(nvars)), c)

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]

        #Add equality constraints of the form Ax = b
        self.prob_.addRows(
            A[:leq_start, :].shape[0],
            b[:leq_start],
            b[:leq_start],
            A[:leq_start, :].getnnz(),
            A[:leq_start, :].indptr,
            A[:leq_start, :].indices,
            A[:leq_start, :].data
        )
        #Add inequality constraints of the form Ax < b
        self.prob_.addRows(
            (leq_end - leq_start),
            [-inf] * (leq_end - leq_start),
            b[leq_start: leq_end],
            A[leq_start: leq_end, :].getnnz(),
            A[leq_start: leq_end, :].indptr,
            A[leq_start: leq_end, :].indices,
            A[leq_start: leq_end, :].data
        )

        #Save file (*.mst. *.sol, etc.)
        if 'write_model' in solver_opts:
            self.prob_.writeModel(solver_opts['write_model'])
        
        #Pass Solver specific options
        for key, value in solver_opts.items():
            self.prob_.setOptionValue(key, value)

        #Call the solver
        solution = {}
        run_status = self.prob_.run()
        if run_status == hp.HighsStatus.kError:
            raise Exception

        info = self.prob_.getInfo()
        solution["status"] = get_status_maps()[self.prob_.getModelStatus()]
        solution["value"] = info.objective_function_value
        solution["primal"] = np.array(self.prob_.getSolution().col_value)
        solution["dual"] = np.array(self.prob_.getSolution().col_dual)
        solution[s.SOLVE_TIME] = self.prob_.getRunTime()
        solution["model"] = self.prob_.getModel()

        if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
            eq_ineq_dual_values = -np.array(self.prob_.getSolution().row_dual)
            solution[s.EQ_DUAL] = eq_ineq_dual_values[0:nrowsEQ]
            solution[s.INEQ_DUAL] = eq_ineq_dual_values[nrowsEQ:]
        
        self.prob_.clear()
        return solution

def get_status_maps():
    """Create status maps from HiGHS to CVXPY
    """
    import highspy as hp

    status_map = {
        hp.HighsModelStatus.kOptimal:                   s.OPTIMAL,
        hp.HighsModelStatus.kInfeasible:                s.INFEASIBLE,
        hp.HighsModelStatus.kUnbounded:                 s.UNBOUNDED,
        hp.HighsModelStatus.kUnboundedOrInfeasible:     s.INFEASIBLE_OR_UNBOUNDED,
        hp.HighsModelStatus.kTimeLimit:                 s.USER_LIMIT,
        hp.HighsModelStatus.kUnknown:                   s.UNKNOWN,
        hp.HighsModelStatus.kSolveError:                s.SOLVER_ERROR
    }

    return status_map