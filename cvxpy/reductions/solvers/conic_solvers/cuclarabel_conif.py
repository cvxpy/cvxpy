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
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


def dims_to_solver_cones(jl, cone_dims):

    jl.seval("""cones = Clarabel.SupportedCone[]""")

    # assume that constraints are presented
    # in the preferred ordering of SCS.

    if cone_dims.zero > 0:
        jl.push_b(jl.cones, jl.Clarabel.ZeroConeT(cone_dims.zero))

    if cone_dims.nonneg > 0:
        jl.push_b(jl.cones, jl.Clarabel.NonnegativeConeT(cone_dims.nonneg))

    for dim in cone_dims.soc:
        jl.push_b(jl.cones, jl.Clarabel.SecondOrderConeT(dim))

    for dim in cone_dims.psd:
        jl.push_b(jl.cones, jl.Clarabel.PSDTriangleConeT(dim))

    for _ in range(cone_dims.exp):
        jl.push_b(jl.cones, jl.Clarabel.ExponentialConeT())

    for pow in cone_dims.p3d:
        jl.push_b(jl.cones, jl.Clarabel.PowerConeT(pow))


class CUCLARABEL(ConicSolver):
    """An interface for the Clarabel solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS \
        + [SOC, ExpCone, PowCone3D]

    STATUS_MAP = {
                    "SOLVED": s.OPTIMAL,
                    "PRIMAL_INFEASIBLE": s.INFEASIBLE,
                    "DUAL_INFEASIBLE": s.UNBOUNDED,
                    "ALMOST_SOLVED": s.OPTIMAL_INACCURATE,
                    "ALMOST_PRIMAL_INFEASIBLE": s.INFEASIBLE_INACCURATE,
                    "Almost_DUAL_INFEASIBLE": s.UNBOUNDED_INACCURATE,
                    "MAX_ITERATIONS": s.USER_LIMIT,
                    "MAX_TIME": s.USER_LIMIT,
                    "NUMERICAL_ERROR": s.SOLVER_ERROR,
                    "INSUFFICIENT_PROGRESS": s.SOLVER_ERROR
                }

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver.
        """
        return 'CUCLARABEL'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        # Does NOT import juliacall because that starts a Julia interpreter
        # and this method is called on CVXPY importing and that's too heavy.
        import cupy  # noqa F401

    def supports_quad_obj(self) -> bool:
        """Clarabel supports quadratic objective with any combination
        of conic constraints.
        """
        return True

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.
        """
        return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        attr = {}
        status = self.STATUS_MAP[str(solution.status)]
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations
        # more detailed statistics here when available
        # attr[s.EXTRA_STATS] = solution.extra.FOO

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[CUCLARABEL.VAR_ID]: np.array(solution.x)
            }
            eq_dual_vars = utilities.get_dual_values(
                np.array(solution.z[:inverse_data[ConicSolver.DIMS].zero]),
                self.extract_dual_value,
                inverse_data[CUCLARABEL.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                np.array(solution.z[inverse_data[ConicSolver.DIMS].zero:]),
                self.extract_dual_value,
                inverse_data[CUCLARABEL.NEQ_CONSTR]
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
            Whether to warm_start Clarabel.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            Clarabel-specific solver options.

        Returns
        -------
        The result returned by a call to clarabel.solve().
        """
        import cupy
        from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
        from juliacall import Main as jl
        jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
        jl.seval('using CUDA, CUDA.CUSPARSE')

        A = data[s.A]
        b = data[s.B]
        q = data[s.C]

        if s.P in data:
            P = data[s.P]
        else:
            nvars = q.size
            P = sp.csr_array((nvars, nvars))

        P = sp.triu(P).tocsr()

        cones = data[ConicSolver.DIMS]

        Pgpu = cucsr_matrix(P)
        qgpu = cupy.array(q)

        Agpu = cucsr_matrix(A)
        bgpu = cupy.array(b)

        if Pgpu.nnz != 0:
            jl.P = jl.Clarabel.cupy_to_cucsrmat(
                jl.Float64, int(Pgpu.data.data.ptr), int(Pgpu.indices.data.ptr),
                int(Pgpu.indptr.data.ptr), *Pgpu.shape, Pgpu.nnz)
        else:
            jl.seval(f"""
            P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {nvars}, {nvars}))
            """)
        jl.q = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(qgpu.data.ptr), qgpu.size)

        jl.A = jl.Clarabel.cupy_to_cucsrmat(
                jl.Float64, int(Agpu.data.data.ptr), int(Agpu.indices.data.ptr),
                int(Agpu.indptr.data.ptr), *Agpu.shape, Agpu.nnz)
        jl.b = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bgpu.data.ptr), bgpu.size)


        dims_to_solver_cones(jl, cones)

        results = jl.seval("""
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        solver   = Clarabel.Solver(P,q,A,b,cones, settings)
        Clarabel.solve!(solver)
        """)
        return results
    
    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["CUCLARABEL"]
