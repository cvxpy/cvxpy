"""
Copyright 2013 Steven Diamond

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

from noncvx_variable import NonCvxVariable
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class Boolean(NonCvxVariable):
    """ A boolean variable. """
    # Sets the initial z value to a matrix of 0.5's.
    def init_z(self):
        self.z.value = np.zeros(self.size) + 0.5

    # All values set rounded to zero or 1.
    def _round(self, matrix):
        return np.around(matrix)

    # Constrain all entries to be the value in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        obj, constraints = super(Boolean, self).canonicalize()
        one = lu.create_const(np.ones(self.size), self.size)
        constraints += [lu.create_geq(obj),
                        lu.create_leq(obj, one)]
        return (obj, constraints)
