"""
"""


import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class PDCS(ConicSolver):
    """An interface for the PDCS solver.
    """
    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone]
    STATUS_MAP = {
        "optimal" : s.OPTIMAL,
        "primal_infeasible_high_acc": s.INFEASIBLE,
        "dual_infeasible_high_acc": s.UNBOUNDED,
        "primal_infeasible_low_acc": s.INFEASIBLE_INACCURATE,
        "dual_infeasible_low_acc": s.UNBOUNDED_INACCURATE,
        "max_iter": s.USER_LIMIT,
        "time_limit": s.USER_LIMIT,
        "numerical_error": s.SOLVER_ERROR,
    }

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]


    def name(self):
        """The name of the solver.
        """
        return 'PDCS'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import cupy
    
    def supports_quad_obj(self) -> bool:
        """PDCS does not support quadratic objective with any combination
        of conic constraints.
        """
        return False
    
    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.
        """
        return utilities.extract_dual_value(result_vec, offset, constraint)
    
    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem give the inverse_data.
        """
        attr = {}
        status = self.STATUS_MAP[str(solution.info.exit_status)]
        attr[s.SOLVE_TIME] = solution.info.time
        attr[s.NUM_ITERS] = solution.info.iter


        if status in s.SOLUTION_PRESENT:
            primal_val = solution.info.pObj
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[PDCS.VAR_ID]: np.array(solution.x.recovered_primal.primal_sol.x)
            }
            eq_dual_vars = utilities.get_dual_values(
                np.array(solution.y.recovered_dual.dual_sol.y[:inverse_data[ConicSolver.DIMS].zero]),
                self.extract_dual_value,
                inverse_data[PDCS.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                np.array(solution.y.recovered_dual.dual_sol.y[inverse_data[ConicSolver.DIMS].zero:]),
                self.extract_dual_value,
                inverse_data[PDCS.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start PDCS.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            PDCS-specific solver options.

        Returns
        -------
        The result returned by a call to PDCS_GPU.solve_with_solver().
        """
        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl
        jl.seval('using Pkg')
        jl.seval("Pkg.activate(\"./PDCS_fork/pdcs_env\")")
        jl.seval('using LinearAlgebra, SparseArrays')
        jl.seval('using CUDA, CUDA.CUSPARSE, SparseMatricesCSR')
        jl.seval('include("./PDCS_fork/src/pdcs_gpu/PDCS_GPU.jl")')
        A = data[s.A]
        b = data[s.B]
        q = data[s.C]
        if s.P in data:
            raise ValueError("PDCS does not support quadratic objective with any combination of conic constraints.")
        
        cones = data[ConicSolver.DIMS]
        cgpu = cupy.array(q)

        Ggpu = cucsr_matrix(A)
        bgpu = cupy.array(b)

        jl.c = jl.PDCS_GPU.cupy_to_cuvector(jl.Float64, int(cgpu.data.ptr), cgpu.size)

        jl.G = jl.PDCS_GPU.cupy_to_cucsrmat(
                jl.Float64, int(Ggpu.data.data.ptr), int(Ggpu.indices.data.ptr),
                int(Ggpu.indptr.data.ptr), *Ggpu.shape, Ggpu.nnz)
        jl.b = jl.PDCS_GPU.cupy_to_cuvector(jl.Float64, int(bgpu.data.ptr), bgpu.size)

        jl.soc = cones.soc
        jl.expG = cones.exp
        jl.m_zero = cones.zero
        jl.m_nonnegative = cones.nonneg
        assert len(cones.p3d) == 0, "PowCone3D is not supported"
        assert len(cones.pnd) == 0, "PowerCone is not supported"
        assert len(cones.psd) == 0, "PSD is not supported"
        jl.seval("""
        m, n = size(G)
        solver = PDCS_GPU.PDCS_GPU_Solver(
            n = n,
            m = m,
            nb = n,
            c = c,
            G = -G,
            h = -b,
            mGzero = m_zero,
            mGnonnegative = m_nonnegative,
            socG = Vector{Integer}(soc),
            rsocG = Vector{Integer}([]),
            expG = expG,
            dual_expG = 0,
            bl = CuArray(ones(n) * -Inf),
            bu = CuArray(ones(n) * Inf),
            soc_x = Vector{Integer}([]),
            rsoc_x = Vector{Integer}([]),
            exp_x = 0,
            dual_exp_x = 0,
            use_preconditioner = true,
            method = :average,
            print_freq = 2000,
            time_limit = 1000.0,
            use_adaptive_restart = true,
            use_adaptive_step_size_weight = true,
            use_resolving = true,
            use_accelerated = false,
            use_aggressive = true,
            verbose = 2,
            rel_tol = 1e-6,
            abs_tol = 1e-6,
            kkt_restart_freq = 2000,
            duality_gap_restart_freq = 2000,
            use_kkt_restart = false,
            use_duality_gap_restart = true,
            logfile_name = nothing,
        )
        """)
        results = jl.PDCS_GPU.solve_with_solver(jl.solver)
        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["PDCS"]