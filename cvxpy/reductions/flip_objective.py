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
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize, Minimize
from cvxpy.problems.attributes import is_minimization
from cvxpy.reductions.inverse_data import InverseData


class FlipObjective(Reduction):

    preconditions = {
        (Problem, is_minimization, False)
    }

    @staticmethod
    def postconditions(problem_type):
        postconditions = set(t for t in problem_type if t != (Problem, is_minimization, False))
        for c in postconditions:
            if c[0] == Maximize:
                postconditions.remove(c)
                postconditions.add((Minimize, c[1], c[2]))
        return postconditions.union({(Problem, is_minimization, True)})

    def apply(self, problem):
        inverse_data = InverseData(problem)
        if type(problem.objective) == Maximize:
            problem = Problem(Minimize(-problem.objective.expr), problem.constraints)
        return problem, inverse_data

    def invert(self, solution, inverse_data):
        new_solution = solution.copy()
        new_solution.opt_val = -solution.opt_val
        return new_solution
