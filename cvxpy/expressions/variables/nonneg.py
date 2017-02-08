"""
Copyright 2017 Steven Diamond

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

from cvxpy.expressions.variables.variable import Variable
import cvxpy.lin_ops.lin_utils as lu


class NonNegative(Variable):
    """A variable constrained to be nonnegative.
    """

    def canonicalize(self):
        """Enforce that var >= 0.
        """
        obj, constr = super(NonNegative, self).canonicalize()
        return (obj, constr + [lu.create_geq(obj)])

    def __repr__(self):
        return "NonNegative(%d, %d)" % self.size

    def is_positive(self):
        """Is the expression positive?
        """
        return True

    def is_negative(self):
        """Is the expression negative?
        """
        return False
