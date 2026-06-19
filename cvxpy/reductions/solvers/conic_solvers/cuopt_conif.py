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
    # cuOpt cannot solve mixed-integer SOC problems (solve_via_data raises for
    # that combination), so MI support must not advertise SOC. Otherwise the
    # solver-selection logic would route MI-SOCPs here and hit a hard
    # SolverError at solve time instead of deferring to a capable backend.
    MI_SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
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
    def _pad_quadratic_objective(Qcsr, num_vars):
        """Pad Q to num_vars x num_vars for cuOpt SOC variable permutation."""
        n = Qcsr.shape[0]
        if n == num_vars:
            return Qcsr
        pad_cols = num_vars - n
        pad_rows = num_vars - n
        return sp.vstack(
            [
                sp.hstack([Qcsr, sp.csr_array((n, pad_cols), dtype=Qcsr.dtype)]),
                sp.csr_array((pad_rows, num_vars), dtype=Qcsr.dtype),
            ],
            format="csr",
        )

    @staticmethod
    def _build_soc_lift_all(Acsr, b, leq_end, soc_dims, n_orig):
        """Build all SOC lifts aux_i + A[row,:]@x = b[row] in one CSR matrix.

        Returns
        -------
        A_lift : csr_array or None
            Shape (sum(soc_dims), n_orig + sum(soc_dims)).
        rhs_lift : ndarray or None
            Equality RHS for each lift row.
        qc_cones : list of (aux_base, constr_len)
            Cones that need a Lorentz QCMATRIX row (constr_len > 1).
        """
        n_soc = sum(soc_dims)
        if n_soc == 0:
            return None, None, []

        num_vars = n_orig + n_soc
        idx_dtype = Acsr.indices.dtype
        soc_row = leq_end
        nnz = 0
        for constr_len in soc_dims:
            for local_i in range(constr_len):
                nnz += 1 + (Acsr.indptr[soc_row + local_i + 1] - Acsr.indptr[soc_row + local_i])
            soc_row += constr_len

        data = np.empty(nnz, dtype=Acsr.dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        indptr = np.empty(n_soc + 1, dtype=Acsr.indptr.dtype)
        rhs = np.empty(n_soc, dtype=np.float64)

        pos = 0
        out_row = 0
        soc_row = leq_end
        aux_offset = 0
        qc_cones = []

        for constr_len in soc_dims:
            aux_base = n_orig + aux_offset
            if constr_len > 1:
                qc_cones.append((aux_base, constr_len))
            for local_i in range(constr_len):
                row = soc_row + local_i
                indptr[out_row] = pos
                indices[pos] = aux_base + local_i
                data[pos] = 1.0
                pos += 1
                start, end = Acsr.indptr[row], Acsr.indptr[row + 1]
                row_nnz = end - start
                if row_nnz:
                    indices[pos:pos + row_nnz] = Acsr.indices[start:end]
                    data[pos:pos + row_nnz] = Acsr.data[start:end]
                    pos += row_nnz
                rhs[out_row] = b[row]
                out_row += 1
            soc_row += constr_len
            aux_offset += constr_len

        indptr[n_soc] = pos
        A_lift = sp.csr_array(
            (data, indices, indptr),
            shape=(n_soc, num_vars),
        )
        return A_lift, rhs, qc_cones

    @staticmethod
    def _build_working_problem(Acsr, B, leq_start, leq_end, soc_dims, n_orig):
        """Assemble A_work and equality row bounds for eq/ineq + SOC lifts."""
        n_soc = sum(soc_dims)
        lower_bounds = np.empty(leq_end + n_soc, dtype=np.float64)
        lower_bounds[:leq_start] = B[:leq_start]
        lower_bounds[leq_start:leq_end] = float("-inf")
        upper_bounds = np.empty(leq_end + n_soc, dtype=np.float64)
        upper_bounds[:leq_end] = B[:leq_end]

        if n_soc == 0:
            return Acsr[:leq_end, :], lower_bounds, upper_bounds, []

        A_lift, rhs_soc, qc_cones = CUOPT._build_soc_lift_all(
            Acsr, B, leq_end, soc_dims, n_orig
        )
        lower_bounds[leq_end:] = rhs_soc
        upper_bounds[leq_end:] = rhs_soc

        A_lin = Acsr[:leq_end, :]
        if leq_end > 0:
            A_pad = sp.hstack(
                [
                    A_lin,
                    sp.csr_array((leq_end, n_soc), dtype=Acsr.dtype),
                ],
                format="csr",
            )
            A_work = sp.vstack([A_pad, A_lift], format="csr")
        else:
            A_work = A_lift

        return A_work, lower_bounds, upper_bounds, qc_cones

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        verbose = verbose or solver_opts.get("solver_verbose", False) in [True, "True", "true"]
        Acsr = sp.csr_array(data[s.A])
        B = data[s.B]
        C = data[s.C].copy()

        Qcsr = None
        if s.P in data:
            Qcsr = sp.csr_array(data[s.P]) / 2

        n_orig = data[s.C].shape[0]
        dims = dims_to_solver_dict(data[s.DIMS])
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        soc_dims = dims.get(s.SOC_DIM, [])
        n_soc = sum(soc_dims)
        num_vars = n_orig + n_soc
        has_soc = n_soc > 0

        variable_types = np.full(num_vars, "C", dtype="U1")
        variable_lower_bounds = data[s.LOWER_BOUNDS]
        variable_upper_bounds = data[s.UPPER_BOUNDS]
        if variable_lower_bounds is None:
            variable_lower_bounds = np.full(n_orig, -np.inf)
        else:
            variable_lower_bounds = variable_lower_bounds.copy()
        if variable_upper_bounds is None:
            variable_upper_bounds = np.full(n_orig, np.inf)
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

        if n_soc > 0:
            C = np.concatenate([C, np.zeros(n_soc, dtype=np.float64)])
            # cuOpt Lorentz QCs require a non-negative lower bound on the cone
            # head (aux index 0 of each SOC block). Tail aux vars stay free.
            soc_lower_bounds = np.full(n_soc, -np.inf, dtype=np.float64)
            aux_offset = 0
            for constr_len in soc_dims:
                soc_lower_bounds[aux_offset] = 0.0
                aux_offset += constr_len
            variable_lower_bounds = np.concatenate(
                [variable_lower_bounds, soc_lower_bounds]
            )
            variable_upper_bounds = np.concatenate(
                [variable_upper_bounds, np.full(n_soc, np.inf, dtype=np.float64)]
            )

        A_work, lower_bounds, upper_bounds, qc_cones = self._build_working_problem(
            Acsr, B, leq_start, leq_end, soc_dims, n_orig
        )

        qc_specs = []
        qc_row_index = leq_end
        for aux_base, constr_len in qc_cones:
            qc_specs.append(
                (
                    qc_row_index,
                    self._lorentz_qcoo(
                        np.arange(aux_base, aux_base + constr_len, dtype=np.int32)
                    ),
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
        # The objective offset is re-applied in invert() (the standard CVXPY
        # contract). cuOpt's get_primal_objective() folds set_objective_offset
        # into the reported value, so passing it here too would double-count it
        # in Solution.opt_val (visible e.g. via partial_optimize). Keep the
        # backend objective offset-free; invert() adds the constant exactly once.
        data_model.set_objective_offset(0.0)
        data_model.set_constraint_lower_bounds(lower_bounds)
        data_model.set_constraint_upper_bounds(upper_bounds)
        if Qcsr is not None:
            # cuOpt SOC conversion permutes variables and requires Q in full
            # CSR form with indptr length num_vars + 1 (see translate_soc.hpp).
            Qcsr = self._pad_quadratic_objective(Qcsr, num_vars)
            data_model.set_quadratic_objective_matrix(
                Qcsr.data, Qcsr.indices, Qcsr.indptr
            )
        data_model.set_variable_lower_bounds(variable_lower_bounds)
        data_model.set_variable_upper_bounds(variable_upper_bounds)
        data_model.set_variable_types(variable_types)

        for qc_row_index, (qv, qr, qc) in qc_specs:
            data_model.add_quadratic_constraint(
                constraint_row_name=f"soc_{qc_row_index}",
                vals=qv,
                rows=qr,
                cols=qc,
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

        primal_full = cuopt_result.get_primal_solution()
        primal = primal_full[:n_orig] if primal_full is not None else np.array([])

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
