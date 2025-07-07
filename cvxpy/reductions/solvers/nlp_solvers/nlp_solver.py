"""
Copyright 2025, the CVXPY developers

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

from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solvers.solver import Solver


class NLPsolver(Solver):
    """
    A non-linear programming (NLP) solver.
    """
    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    IS_MIP = "IS_MIP"

    def accepts(self, problem):
        # can accept everything?
        return True

    def apply(self, problem):
        """
        Construct NLP problem data stored in a dictionary.
        The NLP has the following form

            minimize      f(x)
            subject to    g^l <= g(x) <= g^u
                          x^l <= x <= x^u
        where f and g are non-linear (and possibly non-convex) functions
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        return data, inv_data

    def _prepare_data_and_inv_data(self, problem):
        data = {}
        inverse_data = InverseData(problem)
        data["problem"] = problem

        inverse_data.offset = 0.0
        return problem, data, inverse_data
