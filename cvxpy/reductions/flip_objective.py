"""
Copyright 2017 Robin Verschueren

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

from cvxpy.expressions import cvxtypes
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.reductions.reduction import Reduction


class FlipObjective(Reduction):
    """Flip a minimization objective to a maximization and vice versa.
     """

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        """:math:`\\max(f(x)) = -\\min(-f(x))`

        Parameters
        ----------
        problem : Problem
            The problem whose objective is to be flipped.

        Returns
        -------
        Problem
            A problem with a flipped objective.
        list
            The inverse data.
        """
        is_maximize = type(problem.objective) == Maximize
        objective = Minimize if is_maximize else Maximize
        problem = cvxtypes.problem()(objective(-problem.objective.expr),
                                     problem.constraints)
        return problem, []

    def invert(self, solution, inverse_data):
        """Map the solution of the flipped problem to that of the original.

        Parameters
        ----------
        solution : Solution
            A solution object.
        inverse_data : list
            The inverse data returned by an invocation to apply.

        Returns
        -------
        Solution
            A solution to the original problem.
        """
        if solution.opt_val is not None:
            solution.opt_val = -solution.opt_val
        return solution
