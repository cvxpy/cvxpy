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

from cvxpy.reductions.utilities import REDUCTIONS
from cvxpy.problems.problem_type import ProblemType


class PathFinder(object):

    def __init__(self):
        self.reductions = set(REDUCTIONS)

    def reduction_path(self, current_type, current_path, solver):
        """Depth first search to target solver.
        """
        for reduction in self.applicable_reductions(current_type):
            self.reductions.remove(reduction)
            current_path.append(reduction)
            if reduction == solver:
                return current_path
            # HACK need better way of distinguishing solvers.
            if not hasattr(reduction, 'postconditions'):
                current_path.pop()
                continue
            old_type = current_type
            current_type = ProblemType(reduction.postconditions(current_type))
            candidate_path = self.reduction_path(current_type, current_path, solver)
            if candidate_path is not None:
                return candidate_path
            else:
                self.reductions.add(reduction)
                current_path.pop()
                current_type = old_type
        return None

    def applicable_reductions(self, problem_type):
        return [red for red in self.reductions if problem_type.matches(red.preconditions)]
