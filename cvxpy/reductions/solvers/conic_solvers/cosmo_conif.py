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
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL  # , dims_to_solver_cones
from cvxpy.utilities.citations import CITATION_DICT


class COSMO(CLARABEL):
    """An interface for the COSMO solver."""

    STATUS_MAP = {
        "Solved": s.OPTIMAL,
        "Primal_infeasible": s.INFEASIBLE,
        "Dual_infeasible": s.UNBOUNDED,
        "Max_iter_reached": s.USER_LIMIT,
        "Time_limit_reached": s.USER_LIMIT,
        "Unsolved": s.SOLVER_ERROR,
    }

    def name(self):
        """The name of the solver."""
        return "COSMO"

    def import_solver(self) -> None:
        """Imports the solver."""
        import cosmopy  # noqa F401

    def invert(self, solution, inverse_data):
        import cosmopy

        model: cosmopy.Model = solution

        attr = {}
        status = self.STATUS_MAP[model.get_status()]
        times = model.get_times()
        attr[s.SETUP_TIME] = times["setup_time"]
        attr[s.SOLVE_TIME] = times["solver_time"]
        attr[s.NUM_ITERS] = model.get_iter()
        attr[s.EXTRA_STATS] = times

        if status in s.SOLUTION_PRESENT:
            primal_val = model.get_objective_value()
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: model.get_x()}

            eq_dual_vars = utilities.get_dual_values(
                model.get_y()[: inverse_data[self.DIMS].zero],
                self.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )

            ineq_dual_vars = utilities.get_dual_values(
                model.get_y()[inverse_data[self.DIMS].zero :],
                self.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )

            dual_vars = eq_dual_vars | ineq_dual_vars

            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_opts(verbose, opts, settings=None):
        if settings is None:
            settings = {}

        settings["verbose"] = verbose

        # use_quad_obj is only for canonicalization.
        if "use_quad_obj" in opts:
            del opts["use_quad_obj"]

        settings |= opts

        return settings

    @classmethod
    def dims_to_solver_cones(cls, dims: ConeDims):
        cones = {}

        if dims.zero > 0:
            cones["f"] = dims.zero
        if dims.nonneg > 0:
            cones["l"] = dims.nonneg
        if len(dims.soc) > 0:
            cones["q"] = dims.soc
        if len(dims.psd) > 0:
            cones["s"] = [(d * (d + 1)) // 2 for d in dims.psd]
        if dims.exp > 0:
            cones["ep"] = dims.exp
        if len(dims.p3d) > 0:
            cones["p"] = dims.p3d

        return cones

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start COSMO.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            COSMO-specific solver options.

        Returns
        -------
        The result returned by a call to Model().optimize().
        """
        import cosmopy

        A = data[s.A].astype(np.float64)
        b = data[s.B].astype(np.float64)
        q = data[s.C].astype(np.float64)

        if s.P in data:
            P = data[s.P].astype(np.float64)
        else:
            nvars = q.size
            P = sp.csc_array((nvars, nvars), dtype=np.float64)

        P = sp.csc_matrix(sp.triu(P))
        A = sp.csc_matrix(A)

        # Cosmo expects the indices to be int32
        A.indices = A.indices.astype(np.int32)
        A.indptr = A.indptr.astype(np.int32)
        P.indices = P.indices.astype(np.int32)
        P.indptr = P.indptr.astype(np.int32)

        cones = self.dims_to_solver_cones(data[self.DIMS])

        settings = self.parse_solver_opts(verbose, solver_opts)
        model = cosmopy.Model()
        model.setup(P, q, A, b, cones, **settings)

        if warm_start:
            try:
                old_model = solver_cache[self.name()]
                x = old_model.get_x()
                y = old_model.get_y()
                model.warm_start(x=x, y=y)
            except KeyError:
                pass

        model.optimize()

        if solver_cache is not None:
            solver_cache[self.name()] = model

        return model

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COSMO"]
