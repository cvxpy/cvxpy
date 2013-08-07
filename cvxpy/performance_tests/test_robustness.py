from cvxpy.atoms import *
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrix_utilities as intf
from cvxopt import matrix
import unittest

class TestProblem(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
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

    # # Overriden method to handle lists and lower accuracy.
    # def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTERFACE):
    #     try:
    #         a = list(a)
    #         b = list(b)
    #         for i in range(len(a)):
    #             self.assertAlmostEqual(a[i], b[i])
    #     except Exception:
    #         super(TestProblem, self).assertAlmostEqual(a,b,places=4)

    # Test large expresssions.
    def test_large_expression(self):
        for n in [10, 20, 30, 40]:
            A = matrix(range(n*n), (n,n))
            x = Variable(n,n)
            p = Problem(Minimize(sum(x)), [x == A])
            result = p.solve()
            answer = n*n*(n*n+1)/2 - n*n
            print result - answer
            self.assertAlmostEqual(result, answer)
