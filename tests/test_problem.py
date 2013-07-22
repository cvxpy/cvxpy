from cvxpy.atoms import *
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

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b):
        if isinstance(a, intf.TARGET_MATRIX):
            a = list(a)
        if isinstance(b, intf.TARGET_MATRIX):
            b = list(b)
        if isinstance(a, list) and isinstance(b, list):
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        else:
            super(TestProblem, self).assertAlmostEqual(a,b,places=4)

    # Test the is_dcp method.
    def test_is_dcp(self):
        p = Problem(Minimize(normInf(self.a)))
        self.assertEqual(p.is_dcp(), True)

        p = Problem(Maximize(normInf(self.a)))
        self.assertEqual(p.is_dcp(), False)

    # Test problems involving variables with the same name.
    def test_variable_name_conflict(self):
        var = Variable(name='a')
        p = Problem(Maximize(self.a + var), [var == 2 + self.a, var <= 3])
        result = p.solve()
        self.assertAlmostEqual(result, 4.0)
        self.assertAlmostEqual(self.a.value, 1)
        self.assertAlmostEqual(var.value, 3)

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
        self.assertAlmostEqual(result, 101 + 1.0/6)
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

    # Test vector LP problems.
    def test_vector_lp(self):
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

    # Test matrix LP problems.
    def test_matrix_lp(self):
        T = matrix(1,(2,2))
        p = Problem(Minimize(1), [self.A == T])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertAlmostEqual(self.A.value, T)

        T = matrix(2,(2,3))
        c = matrix([3,4])
        p = Problem(Minimize(1), [self.A >= T*self.C, 
            self.A == self.B, self.C == T.T])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertAlmostEqual(self.A.value, self.B.value)
        self.assertAlmostEqual(self.C.value, T)
        self.assertGreaterEqual(list(self.A.value), list(T*self.C.value))

    # Test problems with normInf
    def test_normInf(self):
        # Constant argument.
        p = Problem(Minimize(normInf(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(normInf(self.a)), [self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Minimize(3*normInf(self.a + 2*self.b) + self.c), 
            [self.a >= 2, self.b <= -1, self.c == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 3)
        self.assertAlmostEqual(self.a.value + 2*self.b.value, 0)
        self.assertAlmostEqual(self.c.value, 3)

        # Maximize
        p = Problem(Maximize(-normInf(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(normInf(self.x - self.z) + 5), 
            [self.x >= [2,3], self.z <= [-1,-4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertAlmostEqual(list(self.x.value)[1] - list(self.z.value)[1], 7)

    # Test problems with norm1
    def test_norm1(self):
        # Constant argument.
        p = Problem(Minimize(norm1(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # Maximize
        p = Problem(Maximize(-norm1(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(norm1(self.x - self.z) + 5), 
            [self.x >= [2,3], self.z <= [-1,-4]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)
        self.assertAlmostEqual(list(self.x.value)[1] - list(self.z.value)[1], 7)

    # Test problems with norm2
    def test_norm2(self):
        # Constant argument.
        p = Problem(Minimize(norm2(-2)))
        result = p.solve()
        self.assertAlmostEqual(result, 2)

        # Scalar arguments.
        p = Problem(Minimize(norm2(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, -2)

        # Maximize
        p = Problem(Maximize(-norm2(self.a)), [self.a <= -2])
        result = p.solve()
        self.assertAlmostEqual(result, -2)
        self.assertAlmostEqual(self.a.value, -2)

        # Vector arguments.
        p = Problem(Minimize(norm2(self.x - self.z) + 5), 
            [self.x >= [2,3], self.z <= [-1,-4]])
        result = p.solve()
        self.assertAlmostEqual(result, 12.6158)
        self.assertAlmostEqual(self.x.value, [2,3])
        self.assertAlmostEqual(self.z.value, [-1,-4])

    # Test combining atoms
    def test_mixed_atoms(self):
        p = Problem(Minimize(norm2(5 + norm1(self.z) 
                                  + norm1(self.x) + 
                                  normInf(self.x - self.z) ) ), 
            [self.x >= [2,3], self.z <= [-1,-4], norm2(self.x + self.z) <= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertAlmostEqual(self.x.value, [2,3])
        self.assertAlmostEqual(self.z.value, [-1,-4])