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

from cvxpy.reductions import REDUCTIONS as reductions
from cvxpy.problems.problem_type import ProblemType


class PathFinder(object):

    def __init__(self):
        self.reductions = set(reductions)

    def reduction_path(self, current_type, current_path):
        """A reduction path is like a stack: the first element (0) is the end of the reduction
        path (e.g. solver), the first reduction to apply to the problem is path.pop().
        """
        if self.is_valid_reduction_path(current_type, current_path):
            return current_path
        for reduction in self.applicable_reductions(current_type):
            self.reductions.remove(reduction)
            current_path.insert(1, reduction)
            old_type = current_type
            if not hasattr(reduction, 'postconditions'):
                current_path.pop(1)
                current_type = old_type
                continue
            current_type = ProblemType(reduction.postconditions(current_type))
            candidate_path = self.reduction_path(current_type, current_path)
            if candidate_path:
                return candidate_path
            else:
                self.reductions.add(reduction)
                current_path.pop(1)
                current_type = old_type

    def is_valid_reduction_path(self, problem_type, reduction_path):
        solver_preconditions = reduction_path[0].preconditions
        return problem_type.matches(solver_preconditions)

    def applicable_reductions(self, problem_type):
        return [red for red in self.reductions if problem_type.matches(red.preconditions)]
