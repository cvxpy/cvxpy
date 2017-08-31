"""
Copyright 2017 Steven Diamond

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
from cvxpy.problems.solvers.scs_intf import SCS


class SUPERSCS(SCS):
    """An interface for the SuperSCS solver.
    """

    def name(self):
        """The name of the solver.
        """
        return s.SUPERSCS

    def import_solver(self):
        """Imports the solver.
        """
        import superscs
        superscs  # For flake8

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Should the previous solver result be used to warm_start?
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import superscs
        data = self.get_problem_data(objective,
                                     constraints,
                                     cached_data)
        # Set the options to be VERBOSE plus any user-specific options.
        solver_opts["verbose"] = verbose
        scs_args = {"c": data[s.C], "A": data[s.A], "b": data[s.B]}
        # If warm_starting, add old primal and dual variables.
        solver_cache = cached_data[self.name()]
        if warm_start and solver_cache.prev_result is not None:
            scs_args["x"] = solver_cache.prev_result["x"]
            scs_args["y"] = solver_cache.prev_result["y"]
            scs_args["s"] = solver_cache.prev_result["s"]

        results_dict = superscs.solve(scs_args, data[s.DIMS], **solver_opts)
        return self.format_results(results_dict, data, cached_data)
