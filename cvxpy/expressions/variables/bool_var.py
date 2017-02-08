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
from cvxpy.constraints.bool_constr import BoolConstr


class Bool(Variable):
    """ A boolean variable. """

    def canonicalize(self):
        """Variable must be boolean.
        """
        obj, constr = super(Bool, self).canonicalize()
        return (obj, constr + [BoolConstr(obj)])

    def __repr__(self):
        """String to recreate the object.
        """
        return "Bool(%d, %d)" % self.size

    def is_positive(self):
        """Is the expression positive?
        """
        return True

    def is_negative(self):
        """Is the expression negative?
        """
        return False
