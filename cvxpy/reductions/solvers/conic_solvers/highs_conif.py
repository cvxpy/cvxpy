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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT


def unpack_highs_options_inplace(solver_opts) -> None:
    # Users can pass options inside a nested dict -- e.g. to circumvent a name clash
    highs_options = solver_opts.pop("highs_options", dict())

    # merge via update(dict(...)) is needed to avoid silently over-writing options
    solver_opts.update(dict(**solver_opts, **highs_options))


class HIGHS(ConicSolver):
    """
    An interface for the HiGHS solver
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    # Map of HiGHS status to CVXPY status.
    STATUS_MAP = {
        "kNotset": s.SOLVER_ERROR,
        "kModelError": s.SOLVER_ERROR,
        "kSolveError": s.SOLVER_ERROR,
        "kOptimal": s.OPTIMAL,
        "kInfeasible": s.INFEASIBLE,
        "kUnboundedOrInfeasible": s.INFEASIBLE_OR_UNBOUNDED,
        "kUnbounded": s.UNBOUNDED,
        "kObjectiveBound": s.USER_LIMIT,
        "kObjectiveTarget": s.USER_LIMIT,
        "kTimeLimit": s.USER_LIMIT,
        "kIterationLimit": s.USER_LIMIT,
        "kSolutionLimit": s.USER_LIMIT,
    }

    def name(self):
        return s.HIGHS

    def import_solver(self) -> None:
        import highspy

        highspy

    def accepts(self, problem) -> bool:
        """Can HiGHS solve the problem?"""
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
        """Returns a new problem and data for inverting the new solution."""

        data, inv_data = super(HIGHS, self).apply(problem)
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data["is_mip"] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, results, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        attr = {
            s.SOLVE_TIME: results["run_time"],
            s.EXTRA_STATS: results["info"],
        }

        # map solver statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(results["model_status"], s.UNKNOWN)
        if status in s.SOLUTION_PRESENT:
            opt_val = results["info"].objective_function_value + inverse_data[s.OFFSET]
            primal_vars = {
                # inverse_data[HIGHS.VAR_ID]: ...
                # I don't understand how the line below works, the other conif solvers have
                # something similar to the commented line above for the "key".
                HIGHS.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(
                    np.array(results["solution"].col_value)
                )
            }
            # add duals if not a MIP.
            dual_vars = None
            if not inverse_data['is_mip']:
                # The dual values are retrieved in the order that the
                # constraints were added in solve_via_data() below. We
                # must be careful to map them to inverse_data[EQ_CONSTR]
                # followed by inverse_data[NEQ_CONSTR] accordingly.
                y = -np.array(results["solution"].row_dual)
                dual_vars = utilities.get_dual_values(
                    y,
                    utilities.extract_dual_value,
                    inverse_data[HIGHS.EQ_CONSTR] + inverse_data[HIGHS.NEQ_CONSTR])

            attr[s.NUM_ITERS] = (
                results["info"].ipm_iteration_count
                + results["info"].crossover_iteration_count
                + results["info"].pdlp_iteration_count
                + results["info"].qp_iteration_count
                + results["info"].simplex_iteration_count
            )
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None
    ):
        """Returns the result of the call to the solver.

        minimize          cx
            subject to    A x <=  b

        in HiGHS, the opt problem format is,

        minimize          cx
            subject to    lboundA <= A x <= uboundA

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Whether to warm_start HiGHS.
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

        # setup problem data
        inf = hp.Highs().inf
        dims = dims_to_solver_dict(data[s.DIMS])

        c = data[s.C]
        # A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        A = data[s.A].tocsc()
        data["Ax"] = A

        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        uboundA = data[s.B]
        data["u"] = uboundA
        lboundA = np.concatenate(
            [data[s.B][:leq_start], -inf * np.ones(data[s.B][leq_start:leq_end].shape)]
        )
        data["l"] = lboundA

        # setup highs model
        model = hp.HighsModel()
        lp = model.lp_
        lp.num_col_ = A.shape[1]
        # assert data['n_var'] == A.shape[1]
        lp.num_row_ = A.shape[0]
        # assert data['n_eq'] + data['n_ineq'] == A.shape[0]

        # offset already applied in invert()
        lp.offset_ = 0
        lp.col_cost_ = c
        lp.row_lower_ = lboundA
        lp.row_upper_ = uboundA

        assert A.format == "csc"
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data

        # Define Variable bounds
        col_lower = -inf * np.ones(shape=lp.num_col_, dtype=c.dtype)
        col_upper = inf * np.ones(shape=lp.num_col_, dtype=c.dtype)
        # update col_lower and col_upper to account for boolean variables,
        # also set integrality_ for boolean or integers variables
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            # note that integrality_ can only be assigned a list
            integrality = [hp.HighsVarType.kContinuous] * lp.num_col_
            if data[s.BOOL_IDX]:
                for ind in data[s.BOOL_IDX]:
                    integrality[ind] = hp.HighsVarType.kInteger
                bool_mask = np.array(data[s.BOOL_IDX], dtype=int)
                col_lower[bool_mask] = 0
                col_upper[bool_mask] = 1
            for ind in data[s.INT_IDX]:
                integrality[ind] = hp.HighsVarType.kInteger
            lp.integrality_ = integrality
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper

        solver = hp.Highs()

        # setup options
        unpack_highs_options_inplace(solver_opts)
        solver.setOptionValue("log_to_console", verbose)
        for name, value in solver_opts.items():
            # note that calling setOptionValue directly on the solver
            # allows one to pass advanced options that aren't available
            # on the HighOptions class (e.g., presolve_rule_off)
            if solver.setOptionValue(name, value) == hp.HighsStatus.kError:
                raise ValueError(
                    f"HIGHS returned status kError for option (name, value): ({name}, {value})"
                )

        solver.passModel(model)

        if warm_start and solver_cache is not None and self.name() in solver_cache:
            old_solver, old_data, old_result = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_result["model_status"], s.SOLVER_ERROR)
            if old_status in s.SOLUTION_PRESENT:
                solver.setSolution(old_result["solution"])

        # initialize and solve problem
        try:
            solver.run()
            results = {
                "solution": solver.getSolution(),
                "basis": solver.getBasis(),
                "info": solver.getInfo(),
                "model_status": solver.getModelStatus().name,
                "run_time": solver.getRunTime(),
            }
        except ValueError as e:
            raise SolverError(e)

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, results)

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["HIGHS"]
