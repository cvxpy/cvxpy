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

from cvxpy.problems.attributes import attributes as problem_attributes
from cvxpy.expressions.attributes import attributes as expression_attributes
from cvxpy.constraints.attributes import attributes as constraint_attributes
from cvxpy.problems.objective_attributes import attributes as objective_attributes


class ProblemAnalyzer(object):

    def __init__(self, problem):
        self.type = []
        self.type.extend(self.analyze(problem, problem_attributes()))
        con_attributes = expression_attributes() + constraint_attributes()
        self.type.extend(attr for c in problem.constraints
                         for attr in self.analyze(c, con_attributes))
        obj_attributes = expression_attributes() + objective_attributes()
        self.type.extend(self.analyze(problem.objective, obj_attributes))

    def analyze(self, item, attributes):
        """Go through all of the attributes matching the type and return a tuple
        (Type, Attribute, Result).
        """
        for attr in attributes:
            yield (type(item), attr, attr(item))

    def matches(self, preconditions):
        """Checks if all preconditions hold for the problem that is being analyzed."""
        for pre in preconditions:
            properties_to_check = (prop for prop in self.type if
                                   issubclass(prop[0], pre[0]) and prop[1] == pre[1])
            if any(prop[2] != pre[2] for prop in properties_to_check):
                return False
        return True
