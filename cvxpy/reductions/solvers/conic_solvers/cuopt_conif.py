"""
Copyright 2025 NVIDIA CORPORATION

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
from cvxpy.constraints import SOC
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
    ConicSolver,
    dims_to_solver_dict,
)
from cvxpy.utilities.citations import CITATION_DICT

# Wrap cuopt imports in an exception handler so that we
# can have them at module level but not break if cuoopt
# is not installed
try:
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_LOG_TO_CONSOLE,
        CUOPT_METHOD,
        CUOPT_PDLP_SOLVER_MODE,
    )
    from cuopt.linear_programming.solver.solver_wrapper import (
        ErrorStatus,
        LPTerminationStatus,
        MILPTerminationStatus,
    )
    from cuopt.linear_programming.solver_settings import (
        PDLPSolverMode,
        SolverMethod,
        SolverSettings,
    )

    cuopt_present = True
except Exception:
    cuopt_present = False


class CUOPT(ConicSolver):
    """An interface to the cuOpt solver"""

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS
    BOUNDED_VARIABLES = True
    REQUIRES_CONSTR = True
    STATUS_MAP_MIP = {}
    STATUS_MAP_LP = {}

    def _get_status_mip(self, cuopt_status):
        STATUS_MAP_MIP = {
            MILPTerminationStatus.NoTermination: s.SOLVER_ERROR,
            MILPTerminationStatus.Optimal: s.OPTIMAL,
            MILPTerminationStatus.FeasibleFound: s.USER_LIMIT,
            MILPTerminationStatus.Infeasible: s.INFEASIBLE,
            MILPTerminationStatus.Unbounded: s.UNBOUNDED,
            MILPTerminationStatus.TimeLimit: s.USER_LIMIT,
            MILPTerminationStatus.UnboundedOrInfeasible: s.INFEASIBLE_OR_UNBOUNDED,
        }
        return STATUS_MAP_MIP[cuopt_status]

    def _get_status_lp(self, cuopt_status):
        STATUS_MAP_LP = {
            LPTerminationStatus.NoTermination: s.SOLVER_ERROR,
            LPTerminationStatus.NumericalError: s.SOLVER_ERROR,
            LPTerminationStatus.Optimal: s.OPTIMAL,
            LPTerminationStatus.PrimalInfeasible: s.INFEASIBLE,
            LPTerminationStatus.DualInfeasible: s.UNBOUNDED,
            LPTerminationStatus.IterationLimit: s.USER_LIMIT,
            LPTerminationStatus.TimeLimit: s.USER_LIMIT,
            LPTerminationStatus.PrimalFeasible: s.USER_LIMIT,
            LPTerminationStatus.UnboundedOrInfeasible: s.INFEASIBLE_OR_UNBOUNDED,
        }
        return STATUS_MAP_LP[cuopt_status]

    def _solver_mode(self, m):
        try:
            if m.isdigit():
                return PDLPSolverMode(int(m))
            return PDLPSolverMode[m]
        except Exception:
            return None

    def _solver_method(self, m):
        try:
            if m.isdigit():
                return SolverMethod(int(m))
            return SolverMethod[m]
        except Exception:
            return None

    def _get_cuopt_parameter_strings(self):
        from cuopt.linear_programming.solver import solver_parameters

        # Get all attributes that start with CUOPT_
        cuopt_attrs = [attr for attr in dir(solver_parameters) if attr.startswith("CUOPT_")]

        # Extract string values
        result = []
        for attr in cuopt_attrs:
            value = getattr(solver_parameters, attr)
            if isinstance(value, str):
                result.append(value)

        return result

    def name(self):
        """The name of the solver."""
        return s.CUOPT

    def import_solver(self) -> None:
        """Imports the solver."""
        if not cuopt_present:
            raise ModuleNotFoundError()

    def supports_quad_obj(self) -> bool:
        """Cuopt supports quadratic objective."""
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(CUOPT, self).apply(problem)

        # Save the objective offset so that it can be set in the solver
        data[s.OFFSET] = inv_data[s.OFFSET]

        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data["lp"] = not (data[s.BOOL_IDX] or data[s.INT_IDX])

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        status = solution.status

        if status in s.SOLUTION_PRESENT:
            dual_vars = None
            opt_val = solution.opt_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution.primal_vars}
            if s.EQ_DUAL in solution.dual_vars and inverse_data["lp"]:
                dual_vars = {}
                if len(inverse_data[self.EQ_CONSTR]) > 0:
                    eq_dual = utilities.get_dual_values(
                        solution.dual_vars[s.EQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.EQ_CONSTR],
                    )
                    dual_vars.update(eq_dual)
                if len(inverse_data[self.NEQ_CONSTR]) > 0:
                    leq_dual = utilities.get_dual_values(
                        solution.dual_vars[s.INEQ_DUAL],
                        utilities.extract_dual_value,
                        inverse_data[self.NEQ_CONSTR],
                    )
                    dual_vars.update(leq_dual)

            return Solution(status, opt_val, primal_vars, dual_vars, solution.attr)
        else:
            return failure_solution(status)

    # Returns a SolverSettings object
    def _get_solver_settings(self, solver_opts, mip, verbose):
        ss = SolverSettings()
        ss.set_parameter(CUOPT_LOG_TO_CONSOLE, verbose)

        # Special handling for the enum value
        if CUOPT_PDLP_SOLVER_MODE in solver_opts:
            m = self._solver_mode(solver_opts[CUOPT_PDLP_SOLVER_MODE])
            if m is not None:
                ss.set_parameter(CUOPT_PDLP_SOLVER_MODE, m)
            solver_opts.pop(CUOPT_PDLP_SOLVER_MODE)

        # Name collision with "method" in cvxpy
        if "solver_method" in solver_opts:
            m = self._solver_method(solver_opts["solver_method"])
            if m is not None:
                ss.set_parameter(CUOPT_METHOD, m)
            solver_opts.pop("solver_method")

        if "optimality" in solver_opts:
            ss.set_optimality_tolerance(solver_opts["optimality"])
            solver_opts.pop("optimality")

        valid = self._get_cuopt_parameter_strings()
        for p, v in solver_opts.items():
            if p in valid:
                ss.set_parameter(p, solver_opts[p])

        return ss

    @staticmethod
    def _lorentz_qcoo(variable_indices):
        """COO for -x_0^2 + sum_{i>0} x_i^2 <= 0 (cuOpt Lorentz / CVXPY SOC)."""
        indices = np.asarray(variable_indices, dtype=np.int32)
        q_values = np.empty(len(indices), dtype=np.float64)
        q_values[0] = -1.0
        q_values[1:] = 1.0
        return q_values, indices.copy(), indices.copy()

    @staticmethod
    def _soc_lift_csr(Acsr, b, rows, num_vars):
        """Build k equalities aux_i + A[row, :] @ x = b[row] (Gurobi-style lift)."""
        k = len(rows)
        data, indices, indptr = [], [], [0]
        rhs = np.empty(k, dtype=np.float64)
        for local_i, row in enumerate(rows):
            cols = [num_vars + local_i]
            coeffs = [1.0]
            start, end = Acsr.indptr[row], Acsr.indptr[row + 1]
            for j, a_ij in zip(Acsr.indices[start:end], Acsr.data[start:end]):
                cols.append(j)
                coeffs.append(a_ij)
            indices.extend(cols)
            data.extend(coeffs)
            indptr.append(len(data))
            rhs[local_i] = b[row]
        A_lift = sp.csr_matrix(
            (data, indices, indptr),
            shape=(k, num_vars + k),
        )
        return A_lift, rhs

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        verbose = verbose or solver_opts.get("solver_verbose", False) in [True, "True", "true"]
        Acsr = data[s.A].tocsr(copy=False)
        B = data[s.B]
        C = data[s.C].copy()

        Qcsr = None
        if s.P in data:
            Qcsr = data[s.P].tocsr() / 2

        n_orig = data[s.C].shape[0]
        num_vars = n_orig
        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        soc_dims = dims.get(s.SOC_DIM, [])
        has_soc = len(soc_dims) > 0

        lower_bounds = np.empty(leq_end)
        lower_bounds[:leq_start] = B[:leq_start]
        lower_bounds[leq_start:leq_end] = float("-inf")
        upper_bounds = B[:leq_end].copy()

        variable_types = np.full(num_vars, "C", dtype="U1")
        variable_lower_bounds = data[s.LOWER_BOUNDS]
        variable_upper_bounds = data[s.UPPER_BOUNDS]
        if variable_lower_bounds is None:
            variable_lower_bounds = np.full(num_vars, -np.inf)
        else:
            variable_lower_bounds = variable_lower_bounds.copy()
        if variable_upper_bounds is None:
            variable_upper_bounds = np.full(num_vars, np.inf)
        else:
            variable_upper_bounds = variable_upper_bounds.copy()

        is_mip = data[s.BOOL_IDX] or data[s.INT_IDX]
        if is_mip:
            variable_types[data[s.BOOL_IDX] + data[s.INT_IDX]] = "I"
            variable_lower_bounds[data[s.BOOL_IDX]] = 0
            variable_upper_bounds[data[s.BOOL_IDX]] = 1

        if has_soc and is_mip:
            raise SolverError(
                "CUOPT does not support mixed-integer problems with SOC constraints."
            )

        A_work = Acsr[:leq_end, :].tocsr(copy=True)
        soc_start = leq_end
        soc_lifts = []

        for constr_len in soc_dims:
            soc_end = soc_start + constr_len
            rows = range(soc_start, soc_end)
            A_lift, rhs_lift = self._soc_lift_csr(Acsr, B, rows, num_vars)
            aux_base = num_vars
            soc_lifts.append((A_lift, rhs_lift, aux_base, constr_len))
            num_vars += constr_len
            C = np.concatenate([C, np.zeros(constr_len, dtype=np.float64)])
            variable_lower_bounds = np.concatenate(
                [variable_lower_bounds, np.zeros(constr_len, dtype=np.float64)]
            )
            variable_upper_bounds = np.concatenate(
                [variable_upper_bounds, np.full(constr_len, np.inf, dtype=np.float64)]
            )
            variable_types = np.concatenate(
                [variable_types, np.full(constr_len, "C", dtype="U1")]
            )
            soc_start = soc_end

        if A_work.shape[1] < num_vars:
            A_work = sp.hstack(
                [
                    A_work,
                    sp.csr_matrix((A_work.shape[0], num_vars - A_work.shape[1])),
                ],
                format="csr",
            )
        for A_lift, rhs_lift, _, _ in soc_lifts:
            if A_lift.shape[1] < num_vars:
                A_lift = sp.hstack(
                    [
                        A_lift,
                        sp.csr_matrix((A_lift.shape[0], num_vars - A_lift.shape[1])),
                    ],
                    format="csr",
                )
            A_work = sp.vstack([A_work, A_lift], format="csr")
            lower_bounds = np.concatenate([lower_bounds, rhs_lift])
            upper_bounds = np.concatenate([upper_bounds, rhs_lift])

        qc_specs = []
        qc_row_index = A_work.shape[0]
        for _, _, aux_base, constr_len in soc_lifts:
            if constr_len > 1:
                qc_specs.append(
                    (
                        qc_row_index,
                        self._lorentz_qcoo(range(aux_base, aux_base + constr_len)),
                    )
                )
                qc_row_index += 1

        from cuopt.linear_programming.data_model import DataModel
        from cuopt.linear_programming.solver import Solve

        data_model = DataModel()
        data_model.set_csr_constraint_matrix(
            A_work.data, A_work.indices, A_work.indptr
        )
        data_model.set_objective_coefficients(C)
        data_model.set_objective_offset(data[s.OFFSET])
        data_model.set_constraint_lower_bounds(lower_bounds)
        data_model.set_constraint_upper_bounds(upper_bounds)
        if Qcsr is not None:
            data_model.set_quadratic_objective_matrix(
                Qcsr.data, Qcsr.indices, Qcsr.indptr
            )
        data_model.set_variable_lower_bounds(variable_lower_bounds)
        data_model.set_variable_upper_bounds(variable_upper_bounds)
        data_model.set_variable_types(variable_types)

        for qc_row_index, (qv, qr, qc) in qc_specs:
            data_model.add_quadratic_constraint(
                constraint_row_index=qc_row_index,
                constraint_row_name=f"soc_{qc_row_index}",
                quadratic_values=qv,
                quadratic_row_indices=qr,
                quadratic_col_indices=qc,
                sense="L",
            )

        ss = self._get_solver_settings(solver_opts, is_mip, verbose)
        if has_soc:
            ss.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)

        cuopt_result = Solve(data_model, ss)

        if verbose:
            print("Termination reason: ", cuopt_result.get_termination_reason())
        if cuopt_result.get_error_status() != ErrorStatus.Success:
            raise SolverError(cuopt_result.get_error_message())

        dual_vars = {}
        if is_mip:
            sol_status = self._get_status_mip(cuopt_result.get_termination_status())
            extra_stats = cuopt_result.get_milp_stats()
            iters = extra_stats["num_simplex_iterations"]
        else:
            if not has_soc:
                d = cuopt_result.get_dual_solution()
                if d is not None:
                    dual_vars[s.EQ_DUAL] = -d[0:leq_start]
                    dual_vars[s.INEQ_DUAL] = -d[leq_start:leq_end]
            sol_status = self._get_status_lp(cuopt_result.get_termination_status())
            extra_stats = cuopt_result.get_lp_stats()
            iters = extra_stats["nb_iterations"]

        primal = cuopt_result.get_primal_solution()[:n_orig]

        return Solution(
            sol_status,
            cuopt_result.get_primal_objective(),
            primal,
            dual_vars,
            attr={
                s.SOLVE_TIME: cuopt_result.get_solve_time(),
                s.NUM_ITERS: iters,
                s.EXTRA_STATS: extra_stats,
            },
        )

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["CUOPT"]
