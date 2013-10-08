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
from cvxpy.expressions.variables import SDPVar
from cvxopt import matrix
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
        self.X = SDPVar(2)
        self.Y = Variable(2,2)
        self.F = matrix([[1,0],[0,-1]], tc='d')

    def test_sdp_problem(self):
        # SDP in objective.
        # ECHU: this creates 4 separate SDPs, one for each elem in the matrix
        obj = Minimize(sum(square(self.X - self.F)))
        p = Problem(obj,[])
        result = p.solve()
        print self.X.value
        print self.F
        self.assertAlmostEqual(result, 1)
        
        self.assertAlmostEqual(self.X.value[0,0], 1)
        self.assertAlmostEqual(self.X.value[0,1], 0)
        self.assertAlmostEqual(self.X.value[1,0], 0)
        self.assertAlmostEqual(self.X.value[1,1], 0)
        
        # SDP in constraint.
        obj = Minimize(sum(square(self.Y - self.F)))
        p = Problem(obj, [self.Y == SDPVar(2)])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        
        self.assertAlmostEqual(self.Y.value[0,0], 1)
        self.assertAlmostEqual(self.Y.value[0,1], 0)
        self.assertAlmostEqual(self.Y.value[1,0], 0)
        self.assertAlmostEqual(self.Y.value[1,1], 0)
        
        
        # Index into semidef.
        # ECHU: this creates 4 separate SDPs, one for each elem in the matrix
        obj = obj = Minimize(abs(self.X[0,0] - 1) + abs(self.X[1,0] - 2))
        p = Problem(obj, [self.X == self.X.T])#[self.X[0,1] == self.X[1,0]])#[self.X == self.X.T])
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        
        self.assertAlmostEqual(self.X.value[0,0], 1)
        self.assertAlmostEqual(self.X.value[0,1], 2)
        self.assertAlmostEqual(self.X.value[1,0], 2)
        self.assertAlmostEqual(self.X.value[1,1], 6.92, places=2)
