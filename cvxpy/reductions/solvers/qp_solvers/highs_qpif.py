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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


def unpack_highs_options_inplace(solver_opts) -> None:
    # Users can pass options inside a nested dict -- e.g. to circumvent a name clash
    highs_options = solver_opts.pop("highs_options", dict())

    # merge via update(dict(...)) is needed to avoid silently over-writing options
    solver_opts.update(dict(**solver_opts, **highs_options))


class HIGHS(QpSolver):
    """QP interface for the HiGHS solver"""

    # Note that HiGHS does not support MIQP but supports MILP
    MIP_CAPABLE = False

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

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        In HiGHS, the QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    L <= A x <= U

        """
        data, inv_data = super(HIGHS, self).apply(problem)
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
                HIGHS.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(
                    np.array(results["solution"].col_value)
                )
            }
            # add duals if not a MIP.
            dual_vars = None
            if not inverse_data[HIGHS.IS_MIP]:
                dual_vars = {HIGHS.DUAL_VAR_ID: -np.array(results["solution"].row_dual)}
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

        minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        in HiGHS, the opt problem format is,

        minimize      1/2 x' P x + q' x
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
        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        data["Ax"] = A
        uboundA = np.concatenate((data[s.B], data[s.G]))
        data["u"] = uboundA
        lboundA = np.concatenate([data[s.B], -inf * np.ones(data[s.G].shape)])
        data["l"] = lboundA

        # setup highs model
        model = hp.HighsModel()
        lp = model.lp_
        lp.num_col_ = A.shape[1]
        assert data["n_var"] == A.shape[1]
        lp.num_row_ = A.shape[0]
        assert data["n_eq"] + data["n_ineq"] == A.shape[0]

        # offset already applied in invert()
        lp.offset_ = 0
        lp.col_cost_ = q
        lp.row_lower_ = lboundA
        lp.row_upper_ = uboundA

        assert A.format == "csc"
        lp.a_matrix_.format_ = hp.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data

        # Define Variable bounds
        lp.col_lower_ = -inf * np.ones(shape=lp.num_col_, dtype=q.dtype)
        lp.col_upper_ = inf * np.ones(shape=lp.num_col_, dtype=q.dtype)

        # note that we count actual nonzeros because
        # parameter values could make the problem linear
        # (i.e., P.count_nonzero() <= P.nnz)
        if P.count_nonzero():
            hessian = model.hessian_
            hessian.dim_ = model.lp_.num_col_
            assert P.format == "csc"
            hessian.format_ = hp.HessianFormat.kSquare
            hessian.start_ = P.indptr
            hessian.index_ = P.indices
            hessian.value_ = P.data

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
