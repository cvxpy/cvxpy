from cvxpy.atoms.abs import abs
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrices as intf
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

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b):
        if isinstance(a, list) and isinstance(b, list):
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        else:
            super(TestProblem, self).assertAlmostEqual(a,b,places=5)

    # Test scalar LP problems.
    def test_scalar_lp(self):
        p = Problem(Minimize(3*self.a), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 6)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Maximize(3*self.a - self.b), 
            [self.a <= 2, self.b == self.a, self.b <= 5])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 2)

        # With a constant in the objective.
        p = Problem(Minimize(3*self.a - self.b + 100), 
            [self.a >= 2, 
             self.b + 5*self.c - 2 == self.a, 
             self.b <= 5 + self.c])
        result = p.solve()
        #self.assertAlmostEqual(result, 101 + 1.0/6)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.b.value, 5-1.0/6)
        self.assertAlmostEqual(self.c.value, -1.0/6)

        # Infeasible problems
        p = Problem(Maximize(self.a), [self.a >= 2])
        result = p.solve()
        self.assertEqual(result, 'dual infeasible')

        p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
        result = p.solve()
        self.assertEqual(result, 'primal infeasible')

    # Test matrix LP problems.
    def test_matrix_lp(self):
        c = matrix([1,2])
        p = Problem(Minimize(c.T*self.x), [self.x >= c])
        result = p.solve()
        self.assertAlmostEqual(result, 5)
        self.assertAlmostEqual(self.x.value, [1,2])

        A = matrix([[3,5],[1,2]])
        I = intf.identity(2)
        p = Problem(Minimize(c.T*self.x + self.a), 
            [A*self.x >= [-1,1],
             4*I*self.z == self.x,
             self.z >= [2,2],
             self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 26.0)
        self.assertAlmostEqual(self.a.value, 2)
        self.assertAlmostEqual(self.x.value, [8,8])
        self.assertAlmostEqual(self.z.value, [2,2])

    # Test problems with abs
    def test_abs(self):
        p = Problem(Minimize(abs(self.a)), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Minimize(3*abs(self.a + 2*self.b) + self.c), 
            [self.a >= 2, self.b <= -1, self.c == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertAlmostEqual(self.a.value + 2*self.b.value, 0)
        self.assertAlmostEqual(self.c.value, 3)