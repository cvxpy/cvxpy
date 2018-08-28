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

from boolean import Boolean
import cvxopt
import numpy as np
from itertools import product

class Choose(Boolean):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(Choose, self).__init__(rows, cols, *args, **kwargs)

    # Sets the initial z value to the expected value of each entry.
    def init_z(self):
        num_entries = float(self.size[0]*self.size[1])
        self.z.value = cvxopt.matrix(num_entries/self.k, self.size, tc='d')

    # The k-largest values are set to 1. The remainder are set to 0.
    def _round(self, matrix):
        indices = product(xrange(self.size[0]), xrange(self.size[1]))
        v_ind = sorted(indices, key=lambda ind: -matrix[ind])
        for ind in v_ind[0:self.k]:
            matrix[ind] = 1
        for ind in v_ind[self.k:]:
            matrix[ind] = 0
        return matrix
