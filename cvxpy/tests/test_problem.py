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

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTERFACE):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestProblem, self).assertAlmostEqual(a,b,places=3)

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
        self.assertEqual(result, 'Dual infeasible')

        p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
        result = p.solve()
        self.assertEqual(result, 'Primal infeasible')

    # Test vector LP problems.
    def test_vector_lp(self):
        c = matrix([1,2])
        p = Problem(Minimize(c.T*self.x), [self.x >= c])
        result = p.solve()
        self.assertAlmostEqual(result, 5)
        self.assertAlmostEqual(self.x.value, [1,2])

        A = matrix([[3,5],[1,2]])
        I = Constant([[1,0],[0,1]])
        p = Problem(Minimize(c.T*self.x + self.a), 
            [A*self.x >= [-1,1],
             4*I*self.z == self.x,
             self.z >= [2,2],
             self.a >= 2])
        result = p.solve()
        self.assertAlmostEqual(result, 26)
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

        # Test variables are dense.
        self.assertEqual(type(self.A.value), self.A.interface.TARGET_MATRIX)

    # Test variable promotion.
    def test_variable_promotion(self):
        p = Problem(Minimize(self.a), [self.x <= self.a, self.x == [1,2]])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.a.value, 2)

        p = Problem(Minimize(self.a), 
            [self.A <= self.a, 
             self.A == [[1,2],[3,4]]
             ])
        result = p.solve()
        self.assertAlmostEqual(result, 4)
        self.assertAlmostEqual(self.a.value, 4)

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

    # Test problems with abs
    def test_abs(self):
        p = Problem(Minimize(sum(abs(self.A))), [-2 >= self.A])
        result = p.solve()
        self.assertAlmostEqual(result, 8)
        self.assertAlmostEqual(self.A.value, [-2,-2,-2,-2])

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

    # Test using a cvxopt dense matrix as the target.
    def test_dense_matrices(self):
        p = Problem(Minimize(norm2(5 + norm1(self.z) 
                                  + norm1(self.x) + 
                                  normInf(self.x - self.z) ) ), 
            [self.x >= [2,3], self.z <= [-1,-4], norm2(self.x + self.z) <= 2],
            target_matrix=matrix)
        result = p.solve()
        self.assertAlmostEqual(result, 22)
        self.assertAlmostEqual(self.x.value, [2,3])
        self.assertAlmostEqual(self.z.value, [-1,-4])

        T = matrix(2,(2,3))
        c = matrix([3,4])
        p = Problem(Minimize(1), [self.A >= T*self.C, 
            self.A == self.B, self.C == T.T], target_matrix=matrix)
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertAlmostEqual(self.A.value, self.B.value)
        self.assertAlmostEqual(self.C.value, T)
        self.assertGreaterEqual(list(self.A.value), list(T*self.C.value))

    # Test recovery of dual variables.
    def test_dual_variables(self):
        p = Problem(Minimize( norm1(self.x + self.z) ), 
            [self.x >= [2,3],
             [[1,2],[3,4]]*self.z == [-1,-4], 
             norm2(self.x + self.z) <= 100])
        result = p.solve()
        self.assertAlmostEqual(result, 4)
        self.assertAlmostEqual(self.x.value, [4,3])
        self.assertAlmostEqual(self.z.value, [-4,1])
        # Dual values
        self.assertAlmostEqual(p.constraints[0].dual, [0, 1])
        self.assertAlmostEqual(p.constraints[1].dual, [-1, 0.5])
        self.assertAlmostEqual(p.constraints[2].dual, 0)

    
        T = matrix(2,(2,3))
        c = matrix([3,4])
        p = Problem(Minimize(1), 
            [self.A >= T*self.C, 
             self.A == self.B, 
             self.C == T.T])
        result = p.solve()
        # Dual values
        self.assertAlmostEqual(p.constraints[0].dual, 4*[0])
        self.assertAlmostEqual(p.constraints[1].dual, 4*[0])
        self.assertAlmostEqual(p.constraints[2].dual, 6*[0])

    # Test problems with indexing.
    def test_indexing(self):
        # Vector variables
        p = Problem(Maximize(self.x[0,0]), [self.x[0,0] <= 2, self.x[1,0] == 3])
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertAlmostEqual(self.x, [2,3])

        n = 10
        A = matrix(range(n*n), (n,n))
        x = Variable(n,n)
        p = Problem(Minimize(sum(x)), [x == A])
        result = p.solve()
        answer = n*n*(n*n+1)/2 - n*n
        self.assertAlmostEqual(result, answer)

        # Matrix variables
        p = Problem(Maximize( sum(self.A[i,i] + self.A[i,1-i] for i in range(2)) ), 
                             [self.A <= [[1,-2],[-3,4]]])
        result = p.solve()
        self.assertAlmostEqual(result, 0)
        self.assertAlmostEqual(self.A, [1,-2,-3,4])

        # Indexing arithmetic expressions.
        exp = [[1,2],[3,4]]*self.z + self.x
        p = Problem(Minimize(exp[1,0]), [self.x == self.z, self.z == [1,2]])
        result = p.solve()
        self.assertAlmostEqual(result, 12)
        self.assertAlmostEqual(self.x.value, self.z.value)

    # Test the vstack atom.
    def test_vstack(self):
        c = matrix(1, (1,5))
        p = Problem(Minimize(c * vstack(self.x, self.y)), 
            [self.x == [1,2],
            self.y == [3,4,5]])
        result = p.solve()
        self.assertAlmostEqual(result, 15)

        c = matrix(1, (2,2))
        p = Problem( Minimize( sum(vstack(self.A, self.C)) ), 
            [self.A >= 2*c,
            self.C == -2])
        result = p.solve()
        self.assertAlmostEqual(result, -4)