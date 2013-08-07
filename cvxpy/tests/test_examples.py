from cvxpy.atoms import *
from cvxpy.expressions.parameter import Parameter
from cvxpy.expressions.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrix_utilities as intf
import cvxpy.interface.numpy_wrapper as np
import unittest

class TestExamples(unittest.TestCase):
    """ Unit tests using example problems. """

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTERFACE):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestExamples, self).assertAlmostEqual(a,b,places=4)

    # Find the largest Euclidean ball in the polyhedron.
    def test_chebyshev_center(self):
        # The goal is to find the largest Euclidean ball (i.e. its center and
        # radius) that lies in a polyhedron described by linear inequalites in this
        # fashion: P = {x : a_i'*x <= b_i, i=1,...,m} where x is in R^2

        # Generate the input data
        a1 = np.matrix("2; 1")
        a2 = np.matrix(" 2; -1")
        a3 = np.matrix("-1;  2")
        a4 = np.matrix("-1; -2")
        b = np.ones([4,1])

        # Create and solve the model
        r = Variable(name='r')
        x_c = Variable(2,name='x_c')
        obj = Maximize(r)
        constraints = [ #TODO have atoms compute values for constants.
            a1.T*x_c + np.linalg.norm(a1)*r <= b[0],
            a2.T*x_c + np.linalg.norm(a2)*r <= b[1],
            a3.T*x_c + np.linalg.norm(a3)*r <= b[2],
            a4.T*x_c + np.linalg.norm(a4)*r <= b[3],
        ]
        
        p = Problem(obj, constraints)
        result = p.solve()
        self.assertAlmostEqual(result, 0.4472)
        self.assertAlmostEqual(r.value, result)
        self.assertAlmostEqual(x_c.value, [0,0])

    # # Risk return tradeoff curve
    # def test_risk_return_tradeoff(self):
    #     from math import sqrt
    #     from cvxopt import matrix
    #     from cvxopt.blas import dot
    #     from cvxopt.solvers import qp, options
    #     import scipy

    #     n = 4
    #     S = matrix( [[ 4e-2,  6e-3, -4e-3,   0.0 ],
    #                  [ 6e-3,  1e-2,  0.0,    0.0 ],
    #                  [-4e-3,  0.0,   2.5e-3, 0.0 ],
    #                  [ 0.0,   0.0,   0.0,    0.0 ]] )
    #     pbar = matrix([.12, .10, .07, .03])

    #     N = 100
    #     # CVXPY
    #     Sroot = numpy.asmatrix(scipy.linalg.sqrtm(S))
    #     x = Variable(n, name='x')
    #     mu = Parameter(name='mu')
    #     mu.value = 1 # TODO Parameter("positive")
    #     objective = Minimize(-pbar*x + mu*quad_over_lin(Sroot*x,1))
    #     constraints = [sum(x) == 1, x >= 0]
    #     p = Problem(objective, constraints)

    #     mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    #     xs = []
    #     for mu_val in mus:
    #         mu.value = mu_val
    #         p.solve()
    #         xs.append(x.value)
    #     returns = [ dot(pbar,x) for x in xs ]
    #     risks = [ sqrt(dot(x, S*x)) for x in xs ]

    #     # QP solver