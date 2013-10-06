"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy import *
from cvxpy.expressions.variables import Semidefinite
import numpy as np
from base_test import BaseTest
import unittest

def diag(X):
    """ Get the diagonal elements of a matrix.
    
        ECHU: Not sure if we implemented this somewhere already.
    """
    for i in X.size[0]:
        yield X[i,i]

def trace(X):
    """ Compute the trace of a matrix.
    
        ECHU: Not sure if we implemented this somewhere already.
    """
    return sum(diag(X))
    

class TestSemidefiniteVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        self.X = Semidefinite(1)
        self.F = 1# np.matrix([[1,0],[0,-1]])

    def test_log_problem(self):
        pass
        # # SDP in objective.
        # obj = Minimize( square(self.X - self.F) )
        # p = Problem(obj,[])
        # result = p.solve()
        # self.assertAlmostEqual(result, 1)
        # print self.x.value
        # self.assertItemsAlmostEqual(self.x.value, [1,math.e])
        # 
        # # Log in constraint.
        # obj = Minimize(sum(self.x))
        # constr = [log(self.x) >= 0, self.x <= [1,1]]
        # p = Problem(obj, constr)
        # result = p.solve()
        # self.assertAlmostEqual(result, 2)
        # self.assertItemsAlmostEqual(self.x.value, [1,1])
        # 
        # # Index into log.
        # obj = Maximize(log(self.x)[1])
        # constr = [self.x <= [1,math.e]]
        # p = Problem(obj,constr)
        # result = p.solve()
        # self.assertAlmostEqual(result, 1)