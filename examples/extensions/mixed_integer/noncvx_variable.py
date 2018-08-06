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

import abc
import cvxpy
import cvxpy.interface as intf
import cvxopt

class NonCvxVariable(cvxpy.Variable):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        super(NonCvxVariable, self).__init__(*args, **kwargs)
        self.noncvx = True
        self.z = cvxpy.Parameter(*self.size)
        self.init_z()
        self.u = cvxpy.Parameter(*self.size)
        self.u.value = cvxopt.matrix(0, self.size, tc='d')

    # Initializes the value of the replicant variable.
    def init_z(self):
        self.z.value = cvxopt.matrix(0, self.size, tc='d')

    # Verify that the matrix has the same dimensions as the variable.
    def validate_matrix(self, matrix):
        if self.size != intf.size(matrix):
            raise Exception(("The argument's dimensions must match "
                             "the variable's dimensions."))

    # Wrapper to validate matrix.
    def round(self, matrix):
        self.validate_matrix(matrix)
        return self._round(matrix)

    # Project the matrix into the space defined by the non-convex constraint.
    # Returns the updated matrix.
    @abc.abstractmethod
    def _round(matrix):
        return NotImplemented

    # Wrapper to validate matrix and update curvature.
    def fix(self, matrix):
        matrix = self.round(matrix)
        return self._fix(matrix)

    # Fix the variable so it obeys the non-convex constraint.
    @abc.abstractmethod
    def _fix(self, matrix):
        return NotImplemented
