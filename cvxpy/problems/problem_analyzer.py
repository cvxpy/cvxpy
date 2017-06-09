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

import operator


class ProblemAnalyzer(object):

    attributes_to_check = {'is_affine', 'is_quadratic', 'is_qpwa'}

    def __init__(self, problem):
        self.type = []
        self.type.extend(attr for con in problem.constraints for attr in self.analyze(con))
        self.type.extend(self.analyze(problem.objective))

    def analyze(self, item):
        """Go through all of the attributes and return an attribute if it holds for the given
        expression.
        """
        expr = item.expr
        for attr in self.attributes_to_check:
            method = operator.methodcaller(attr)
            if method(expr):
                yield (type(item), attr)

    def check(self, preconditions):
        """Checks if all preconditions hold for the problem that is being analyzed."""
        attribute_types = set(attr[0] for attr in self.type)
        preconditions_to_check = (pre for pre in preconditions if pre[0] in attribute_types)
        if all(pre in self.type for pre in preconditions_to_check):
            return True
        return False
