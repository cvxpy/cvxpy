"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren, 2017 Akshay Agrawal

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

from cvxpy.reductions.solvers.conic_solvers import scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
import cvxpy.settings as s


class DIFFCP(scs_conif.SCS):
    """An interface for the DIFFCP solver, a differentiable wrapper of SCS.
    """

    def name(self):
        """The name of the solver.
        """
        return s.DIFFCP

    def import_solver(self):
        """Imports the solver.
        """
        import diffcp
        patch_version = int(diffcp.__version__.split('.')[2])
        if patch_version < 7:
            raise ImportError("diffcp >= 1.0.7 is required")

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start diffcp.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            SCS-specific solver options.

        Returns
        -------
        The result returned by a call to scs.solve().
        """
        import diffcp
        A = data[s.A]
        b = data[s.B]
        c = data[s.C]
        cones = scs_conif.dims_to_solver_dict(data[ConicSolver.DIMS])
        # Default to eps = 1e-4 instead of 1e-3.
        solver_opts['eps'] = solver_opts.get('eps', 1e-4)
        results = diffcp.solve_and_derivative_internal(
            A, b, c, cones, verbose=verbose, raise_on_error=False,
            **solver_opts)
        if solver_cache is not None:
            solver_cache[self.name()] = results
        return results
