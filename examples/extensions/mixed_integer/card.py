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
import cvxpy.interface.matrix_utilities as intf
from itertools import product

class Card(NonCvxVariable):
    """ A variable with constrained cardinality. """
    # k - the maximum cardinality of the variable.
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(Card, self).__init__(rows, cols, *args, **kwargs)

    # All values except k-largest (by magnitude) set to zero.
    def _round(self, matrix):
        indices = product(xrange(self.size[0]), xrange(self.size[1]))
        v_ind = sorted(indices, key=lambda ind: -abs(matrix[ind]))
        for ind in v_ind[self.k:]:
           matrix[ind] = 0
        return matrix

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        constraints = []
        rows,cols = intf.size(matrix)
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == 0:
                    constraints.append(self[i, j] == 0)
        return constraints
