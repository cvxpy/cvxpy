"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms import *
from cvxpy.expressions.constants import Parameter
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Variable
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

    # Tests examples from the README.
    def test_readme_examples(self):
        import cvxopt
        import numpy
        
        # Problem data.
        m = 30
        n = 20
        A = cvxopt.normal(m,n)
        b = cvxopt.normal(m)

        # Construct the problem.
        x = Variable(n)
        objective = Minimize(sum(square(A*x - b)))
        constraints = [0 <= x, x <= 1]
        p = Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        result = p.solve()
        # The optimal value for x is stored in x.value.
        print x.value
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print constraints[0].dual_value

        ####################################################

        # Scalar variable.
        a = Variable()

        # Column vector variable of length 5.
        x = Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = Variable(4,7)

        ####################################################

        # Positive scalar parameter.
        m = Parameter(sign="positive")

        # Column vector parameter with unknown sign (by default).
        c = Parameter(5)

        # Matrix parameter with negative entries.
        G = Parameter(4,7,sign="negative")

        # Assigns a constant value to G.
        G.value = -numpy.ones((4,7))

        ####################################################
        a = Variable()
        x = Variable(5)

        # expr is an Expression object after each assignment.
        expr = 2*x
        expr = expr - a
        expr = sum(expr) + norm2(x)

        ####################################################

        import numpy as np
        import cvxopt
        from multiprocessing import Pool

        # Problem data.
        n = 10
        m = 5
        A = cvxopt.normal(n,m)
        b = cvxopt.normal(n)
        gamma = Parameter(sign="positive")

        # Construct the problem.
        x = Variable(m)
        objective = Minimize(sum(square(A*x - b)) + gamma*norm1(x))
        p = Problem(objective)

        # Assign a value to gamma and find the optimal x.
        def get_x(gamma_value):
            gamma.value = gamma_value
            result = p.solve()
            return x.value

        gammas = np.logspace(-1, 2, num=2)
        # Serial computation.
        x_values = [get_x(value) for value in gammas]

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