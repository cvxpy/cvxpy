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

from cvxpy.reductions import Reduction
from cvxpy.problems.problem_analyzer import ProblemAnalyzer
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.attributes import is_minimization


class FlipObjective(Reduction):

    preconditions = {
        (Problem, is_minimization, False)
    }

    def accepts(self, problem):
        return ProblemAnalyzer(problem).matches(self.preconditions)

    def apply(self, problem):
        if type(problem.objective) == Maximize:
            return Problem(Minimize(-problem.objective.expr), problem.constraints)
        return problem

    def postconditions(self, problem):
        return (Problem, is_minimization, True)
