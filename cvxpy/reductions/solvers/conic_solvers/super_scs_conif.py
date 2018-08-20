"""
Copyright 2018 Riley Murray

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

import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict, SCS


class SuperSCS(SCS):

    DEFAULT_SETTINGS = {'use_indirect': False, 'eps': 1e-8, 'max_iters': 10000}

    def name(self):
        return s.SUPER_SCS

    def import_solver(self):
        import superscs
        superscs  # For flake8

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        """Returns the result of the call to SuperSCS.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start SuperSCS.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            SuperSCS-specific options.

        Returns
        -------
        The result returned by a call to superscs.solve().
        """
        import superscs
        args = {"A": data[s.A], "b": data[s.B], "c": data[s.C]}
        if warm_start and solver_cache is not None and \
           self.name in solver_cache:
            args["x"] = solver_cache[self.name()]["x"]
            args["y"] = solver_cache[self.name()]["y"]
            args["s"] = solver_cache[self.name()]["s"]
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])
        # settings
        user_opts = list(solver_opts.keys())
        for k in list(SuperSCS.DEFAULT_SETTINGS.keys()):
            if k not in user_opts:
                solver_opts[k] = SuperSCS.DEFAULT_SETTINGS[k]
        results = superscs.solve(
            args,
            cones,
            verbose=verbose,
            **solver_opts)
        if solver_cache is not None:
            solver_cache[self.name()] = results
        return results
