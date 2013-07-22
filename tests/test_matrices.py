from cvxpy.atoms import *
from cvxpy.expressions.expression import *
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrices as intf
import cvxpy.interface.numpy_wrapper as numpy
from cvxopt import matrix
import unittest

class TestMatrices(unittest.TestCase):
    """ Unit tests for testing different forms of matrices as constants. """
    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Test numpy arrays
    def test_numpy_arrays(self):
        # Vector
        v = numpy.arange(2).reshape((2,1))
        self.assertEquals((self.x + v).size(), (2,1))
        self.assertEquals((v + self.x).size(), (2,1))
        self.assertEquals((self.x - v).size(), (2,1))
        self.assertEquals((v - self.x).size(), (2,1))
        # Matrix
        A = numpy.arange(8).reshape((4,2))
        self.assertEquals((A*self.x).size(), (4,1))

    # Test numpy matrices
    def test_numpy_matrices(self):
        # Vector
        v = numpy.matrix( numpy.arange(2).reshape((2,1)) )
        self.assertEquals((self.x + v).size(), (2,1))
        self.assertEquals((v + self.x).size(), (2,1))
        self.assertEquals((self.x - v).size(), (2,1))
        self.assertEquals((v - self.x).size(), (2,1))
        # Matrix
        A = numpy.matrix( numpy.arange(8).reshape((4,2)) )
        self.assertEquals((A*self.x).size(), (4,1))