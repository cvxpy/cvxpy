"""
Copyright 2017 Robin Verschueren

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.expressions import cvxtypes
from cvxpy.reductions import Reduction
from cvxpy.problems.objective import Maximize, Minimize


class FlipObjective(Reduction):
    """Flip a minimization objective to a maximization."""

    def accepts(self, problem):
        return True

    def apply(self, problem):
        is_maximize = type(problem.objective) == Maximize
        if is_maximize:
            problem = cvxtypes.problem()(Minimize(-problem.objective.expr),
                                         problem.constraints)
        return problem, is_maximize

    def invert(self, solution, is_maximize):
        if is_maximize and solution.opt_val is not None:
            solution.opt_val = -solution.opt_val
        return solution
