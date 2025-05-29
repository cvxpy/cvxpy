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

from re import compile

import numpy as np

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT

PYTHON_LIST_SLICE_PATTERN = compile(r", \d+:\d+")
VALID_COLUMN_NAME_PATTERN = compile(
    r"^(?!st$|bounds$|min$|max$|bin$|binary$|gen$|semi$|end$)[a-df-zA-DF-Z\"!#$%&/}{,;?@_‘’'`|~]{1}[a-zA-Z0-9\"!#$%&/}{,;?@_‘’'`|~.=()<>[\]]{,254}$"
)
INVALID_COLUMN_NAME_MESSAGE_TEMPLATE = (
    "Invalid column name: {name}"
    "\nA column name:"
    "\n- must not be equal to one of the keywords: st, bounds, min, max, bin, binary, gen, semi or end"
    "\n- must not begin with a number, the letter e or E or any of the following characters: .=()<>[]"
    "\n- must be alphanumeric (a-z, A-Z, 0-9) or one of these symbols: \"!#$%&/}}{{,;?@_‘’'`|~.=()<>[]"
    "\n- must be no longer than 255 characters."
)


def validate_column_name(name: str) -> bool:
    """Check if the name is a valid column name."""
    if not VALID_COLUMN_NAME_PATTERN.match(name):
        raise ValueError(INVALID_COLUMN_NAME_MESSAGE_TEMPLATE.format(name=name))


def strip_column_name_of_python_list_slice_notation(name: str) -> str:
    """Strip python list slice notation -- i.e., the part after the comma in [0, 0:#]
    - space and colon characters are not allowed in column names and
    - 0:# part is not needed in X[0, 0:#] because we label X[0][0] ... X[0][#] individually
    """
    return PYTHON_LIST_SLICE_PATTERN.sub("", name)


def collect_column_names(variable, column_names):
    """Recursively collect variable names."""
    if variable.ndim == 0:  # scalar
        column_names.append(variable.name())
    elif variable.ndim == 1:  # simple array
        var_name_prefix = strip_column_name_of_python_list_slice_notation(variable.name())
        column_names.extend([f"{var_name_prefix}[{v}]" for v in range(variable.size)])
    else:  # multi-dimensional array
        for var in variable:
            collect_column_names(var, column_names)  # recursive call
    # Checking the validity of only the last inserted name is sufficient because all var
    # names are derived from the same var name prefix and the last one is the longest
    validate_column_name(column_names[-1])


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
                HIGHS.VAR_ID: intf.DEFAULT_INTF.const_to_matrix(
                    np.array(results["solution"].col_value)
                )
            }
            # add duals if not a MIP.
            dual_vars = None
            if not inverse_data["is_mip"]:
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

        # setup options
        unpack_highs_options_inplace(solver_opts)
        options = hp.HighsOptions()
        options.log_to_console = verbose
        for key, value in solver_opts.items():
            setattr(options, key, value)

        if options.write_model_file:
            # TODO: Names can be collected upstream more systematically
            # (or in the parent class) to be used by all solvers.
            column_names = []
            for variable in data[s.PARAM_PROB].variables:
                # NOTE: variable.variable_of_provenance() is a bit of a hack
                # to make sure that auto generated vars are named correctly -- nonneg=True etc.
                variable = variable.variable_of_provenance() or variable
                collect_column_names(variable, column_names)
            lp.col_names_ = column_names

        solver = hp.Highs()
        solver.passOptions(options)
        solver.passModel(model)

        if options.write_model_file:
            # TODO: This part can be removed once the following HiGS PR is released:
            # https://github.com/ERGO-Code/HiGHS/pull/2274
            solver.writeModel(options.write_model_file)

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
