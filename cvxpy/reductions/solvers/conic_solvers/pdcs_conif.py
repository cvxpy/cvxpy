"""
Copyright 2022, the CVXPY Authors

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

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone
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
        import cupy  # noqa F401
    
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
            Options can be passed in Python-native types (bool, int, float, str, etc.); 
            booleans will be converted to "true"/"false" strings for Julia automatically.

        Returns
        -------
        The result returned by a call to PDCS_GPU.solve_with_solver().
        """

        # Helper to preprocess solver_opts so that bools and ints are stringified as Julia expects
        def normalize_opts(opts):
            norm_opts = {}
            for k, v in opts.items():
                if isinstance(v, bool):
                    norm_opts[k] = "true" if v else "false"
                elif isinstance(v, int) and k != "verbose":
                    # verbose can be left as int for logging level
                    norm_opts[k] = str(v)
                else:
                    norm_opts[k] = v
            return norm_opts

        # update default options first (as Python types)
        solver_opts.setdefault("abs_tol", 1e-6)
        solver_opts.setdefault("rel_tol", 1e-6)
        solver_opts.setdefault("logfile", "nothing")
        solver_opts.setdefault("time_limit_secs", 1000.0)
        solver_opts.setdefault("use_scaling", True)
        solver_opts.setdefault("rescaling_method", "ruiz_pock_chambolle")
        solver_opts.setdefault("use_adaptive_restart", True)
        solver_opts.setdefault("use_adaptive_step_size_weight", True)
        solver_opts.setdefault("use_resolving", True)
        solver_opts.setdefault("use_accelerated", False)
        solver_opts.setdefault("use_aggressive", True)
        solver_opts.setdefault("print_freq", 2000)
        solver_opts.setdefault("kkt_restart_freq", 2000)
        solver_opts.setdefault("duality_gap_restart_freq", 2000)
        solver_opts.setdefault("use_kkt_restart", False)
        solver_opts.setdefault("use_duality_gap_restart", True)
        solver_opts.setdefault("use_preconditioner", True)
        solver_opts.setdefault("method", "average")
        solver_opts.setdefault("julia_env", "placeholder")

        # handle verbose pythonically & normalize
        solver_opts["verbose"] = 1 if verbose else 0

        # Convert pythonic boolean/integer options to Julia-friendly strings where needed
        solver_opts = normalize_opts(solver_opts)
        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl

        if solver_opts["julia_env"] != "placeholder":
            jl.seval("using Pkg")
            jl.seval(f"Pkg.activate(\"{solver_opts['julia_env']}\")")
        # jl.seval(f"using Pkg")
        # jl.seval(f"Pkg.develop(path=\"./PDCS\")")
        # jl.seval(f"Pkg.resolve()")
        jl.seval('using CUDA, CUDA.CUSPARSE')
        jl.seval('using PDCS: PDCS_GPU, PDCS_CPU')
        A = data[s.A]
        b = data[s.B]
        q = data[s.C]
        if s.P in data:
            raise ValueError("PDCS does not support quadratic objective.")
        
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
        jl.seval(f"""
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
            socG = Vector{{Integer}}(soc),
            rsocG = Vector{{Integer}}([]),
            expG = expG,
            dual_expG = 0,
            bl = CuArray(ones(n) * -Inf),
            bu = CuArray(ones(n) * Inf),
            soc_x = Vector{{Integer}}([]),
            rsoc_x = Vector{{Integer}}([]),
            exp_x = 0,
            dual_exp_x = 0,
            use_preconditioner = {solver_opts["use_preconditioner"]},
            method = :{solver_opts["method"]},
            print_freq = {solver_opts["print_freq"]},
            time_limit = {solver_opts["time_limit_secs"]},
            use_adaptive_restart = {solver_opts["use_adaptive_restart"]},
            use_adaptive_step_size_weight = {solver_opts["use_adaptive_step_size_weight"]},
            use_resolving = {solver_opts["use_resolving"]},
            use_accelerated = {solver_opts["use_accelerated"]},
            use_aggressive = {solver_opts["use_aggressive"]},
            verbose = {solver_opts["verbose"]},
            rel_tol = {solver_opts["rel_tol"]},
            abs_tol = {solver_opts["abs_tol"]},
            kkt_restart_freq = {solver_opts["kkt_restart_freq"]},
            duality_gap_restart_freq = {solver_opts["duality_gap_restart_freq"]},
            use_kkt_restart = {solver_opts["use_kkt_restart"]},
            use_duality_gap_restart = {solver_opts["use_duality_gap_restart"]},
            logfile_name = {solver_opts["logfile"]},
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