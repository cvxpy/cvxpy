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

import cvxpy as cp
import cvxpy.interface as intf
import numpy as np
from base_test import BaseTest
import cvxopt
import numbers

class TestExamples(BaseTest):
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
        r = cp.Variable(name='r')
        x_c = cp.Variable(2,name='x_c')
        obj = cp.Maximize(r)
        constraints = [ #TODO have atoms compute values for constants.
            a1.T*x_c + np.linalg.norm(a1)*r <= b[0],
            a2.T*x_c + np.linalg.norm(a2)*r <= b[1],
            a3.T*x_c + np.linalg.norm(a3)*r <= b[2],
            a4.T*x_c + np.linalg.norm(a4)*r <= b[3],
        ]

        p = cp.Problem(obj, constraints)
        result = p.solve()
        self.assertAlmostEqual(result, 0.4472)
        self.assertAlmostEqual(r.value, result)
        self.assertItemsAlmostEqual(x_c.value, [0,0])

    # Test issue with numpy scalars.
    def test_numpy_scalars(self):
        n = 6
        eps = 1e-6
        cvxopt.setseed(10)
        P0 = cvxopt.normal(n, n)
        eye = cvxopt.spmatrix(1.0, range(n), range(n))
        P0 = P0.T * P0 + eps * eye

        print P0

        P1 = cvxopt.normal(n, n)
        P1 = P1.T*P1
        P2 = cvxopt.normal(n, n)
        P2 = P2.T*P2
        P3 = cvxopt.normal(n, n)
        P3 = P3.T*P3

        q0 = cvxopt.normal(n, 1)
        q1 = cvxopt.normal(n, 1)
        q2 = cvxopt.normal(n, 1)
        q3 = cvxopt.normal(n, 1)

        r0 = cvxopt.normal(1, 1)
        r1 = cvxopt.normal(1, 1)
        r2 = cvxopt.normal(1, 1)
        r3 = cvxopt.normal(1, 1)

        slack = cp.Variable()
        # Form the problem
        x = cp.Variable(n)
        objective = cp.Minimize( 0.5*cp.quad_form(x,P0) + q0.T*x + r0 + slack)
        constraints = [0.5*cp.quad_form(x,P1) + q1.T*x + r1 <= slack,
                       0.5*cp.quad_form(x,P2) + q2.T*x + r2 <= slack,
                       0.5*cp.quad_form(x,P3) + q3.T*x + r3 <= slack,
        ]

        # We now find the primal result and compare it to the dual result
        # to check if strong duality holds i.e. the duality gap is effectively zero
        p = cp.Problem(objective, constraints)
        primal_result = p.solve()

        # Note that since our data is random, we may need to run this program multiple times to get a feasible primal
        # When feasible, we can print out the following values
        print x.value # solution
        lam1 = constraints[0].dual_value
        lam2 = constraints[1].dual_value
        lam3 = constraints[2].dual_value
        print type(lam1)

        P_lam = P0 + lam1*P1 + lam2*P2 + lam3*P3
        q_lam = q0 + lam1*q1 + lam2*q2 + lam3*q3
        r_lam = r0 + lam1*r1 + lam2*r2 + lam3*r3
        dual_result = -0.5*q_lam.T.dot(P_lam).dot(q_lam) + r_lam
        print dual_result.shape
        self.assertEquals(intf.size(dual_result), (1,1))

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
        x = cp.Variable(n)
        objective = cp.Minimize(sum(cp.square(A*x - b)))
        constraints = [0 <= x, x <= 1]
        p = cp.Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        result = p.solve()
        # The optimal value for x is stored in x.value.
        print x.value
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print constraints[0].dual_value

        ####################################################

        # Scalar variable.
        a = cp.Variable()

        # Column vector variable of length 5.
        x = cp.Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = cp.Variable(4, 7)

        ####################################################

        # Positive scalar parameter.
        m = cp.Parameter(sign="positive")

        # Column vector parameter with unknown sign (by default).
        c = cp.Parameter(5)

        # Matrix parameter with negative entries.
        G = cp.Parameter(4, 7, sign="negative")

        # Assigns a constant value to G.
        G.value = -numpy.ones((4, 7))

        # Raises an error for assigning a value with invalid sign.
        with self.assertRaises(Exception) as cm:
            G.value = numpy.ones((4,7))
        self.assertEqual(str(cm.exception), "Invalid sign for Parameter value.")

        ####################################################
        a = cp.Variable()
        x = cp.Variable(5)

        # expr is an Expression object after each assignment.
        expr = 2*x
        expr = expr - a
        expr = sum(expr) + cp.norm(x, 2)

        ####################################################

        import numpy as np
        import cvxopt
        from multiprocessing import Pool

        # Problem data.
        n = 10
        m = 5
        A = cvxopt.normal(n,m)
        b = cvxopt.normal(n)
        gamma = cp.Parameter(sign="positive")

        # Construct the problem.
        x = cp.Variable(m)
        objective = cp.Minimize(sum(cp.square(A*x - b)) + gamma*cp.norm(x, 1))
        p = cp.Problem(objective)

        # Assign a value to gamma and find the optimal x.
        def get_x(gamma_value):
            gamma.value = gamma_value
            result = p.solve()
            return x.value

        gammas = np.logspace(-1, 2, num=2)
        # Serial computation.
        x_values = [get_x(value) for value in gammas]

        ####################################################
        n = 10

        mu = cvxopt.normal(1, n)
        sigma = cvxopt.normal(n,n)
        sigma = sigma.T*sigma
        gamma = cp.Parameter(sign="positive")
        gamma.value = 1
        x = cp.Variable(n)

        # Constants:
        # mu is the vector of expected returns.
        # sigma is the covariance matrix.
        # gamma is a Parameter that trades off risk and return.

        # Variables:
        # x is a vector of stock holdings as fractions of total assets.

        expected_return = mu*x
        risk = cp.quad_form(x, sigma)

        objective = cp.Maximize(expected_return - gamma*risk)
        p = cp.Problem(objective, [sum(x) == 1])
        result = p.solve()

        # The optimal expected return.
        print expected_return.value

        # The optimal risk.
        print risk.value

        ###########################################

        N = 50
        M = 40
        n = 10
        data = []
        for i in range(N):
            data += [(1, cvxopt.normal(n, mean=1.0, std=2.0))]
        for i in range(M):
            data += [(-1, cvxopt.normal(n, mean=-1.0, std=2.0))]

        # Construct problem.
        gamma = cp.Parameter(sign="positive")
        gamma.value = 0.1
        # 'a' is a variable constrained to have at most 6 non-zero entries.
        a = cp.Variable(n)#mi.SparseVar(n, nonzeros=6)
        b = cp.Variable()

        slack = [cp.pos(1 - label*(sample.T*a - b)) for (label, sample) in data]
        objective = cp.Minimize(cp.norm(a, 2) + gamma*sum(slack))
        p = cp.Problem(objective)
        # Extensions can attach new solve methods to the CVXPY Problem class.
        #p.solve(method="admm")
        p.solve()

        # Count misclassifications.
        errors = 0
        for label, sample in data:
            if label*(sample.T*a - b).value < 0:
                errors += 1

        print "%s misclassifications" % errors
        print a.value
        print b.value

    def test_log_det(self):
        # TODO
        # Generate data
        x = np.matrix("0.55  0.0;"
                      "0.25  0.35;"
                      "-0.2   0.2;"
                      "-0.25 -0.1;"
                      "-0.0  -0.3;"
                      "0.4  -0.2").T
        (n, m) = x.shape

        # Create and solve the model
        A = cp.Variable(n, n);
        b = cp.Variable(n);
        obj = cp.Maximize( cp.log_det(A) )
        constraints = []
        for i in range(m):
            constraints.append( cp.norm(A*x[:, i] + b) <= 1 )
        p = cp.Problem(obj, constraints)
        result = p.solve()
        self.assertAlmostEqual(result, 1.9746)

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
    #     x = cp.Variable(n, name='x')
    #     mu = cp.Parameter(name='mu')
    #     mu.value = 1 # TODO cp.Parameter("positive")
    #     objective = cp.Minimize(-pbar*x + mu*quad_over_lin(Sroot*x,1))
    #     constraints = [sum(x) == 1, x >= 0]
    #     p = cp.Problem(objective, constraints)

    #     mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    #     xs = []
    #     for mu_val in mus:
    #         mu.value = mu_val
    #         p.solve()
    #         xs.append(x.value)
    #     returns = [ dot(pbar,x) for x in xs ]
    #     risks = [ sqrt(dot(x, S*x)) for x in xs ]

    #     # QP solver