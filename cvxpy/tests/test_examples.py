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

from __future__ import print_function
from cvxpy import *
import cvxpy.interface as intf
import numpy as np
from cvxpy.tests.base_test import BaseTest
import cvxopt
import numbers

class TestExamples(BaseTest):
    """ Unit tests using example problems. """

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
        b = np.ones(4)

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
        self.assertAlmostEqual(result, 0.447214)
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

        print(P0)

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

        slack = Variable()
        # Form the problem
        x = Variable(n)
        objective = Minimize( 0.5*quad_form(x,P0) + q0.T*x + r0 + slack)
        constraints = [0.5*quad_form(x,P1) + q1.T*x + r1 <= slack,
                       0.5*quad_form(x,P2) + q2.T*x + r2 <= slack,
                       0.5*quad_form(x,P3) + q3.T*x + r3 <= slack,
        ]

        # We now find the primal result and compare it to the dual result
        # to check if strong duality holds i.e. the duality gap is effectively zero
        p = Problem(objective, constraints)
        primal_result = p.solve()

        # Note that since our data is random, we may need to run this program multiple times to get a feasible primal
        # When feasible, we can print out the following values
        print(x.value) # solution
        lam1 = constraints[0].dual_value
        lam2 = constraints[1].dual_value
        lam3 = constraints[2].dual_value
        print(type(lam1))

        P_lam = P0 + lam1*P1 + lam2*P2 + lam3*P3
        q_lam = q0 + lam1*q1 + lam2*q2 + lam3*q3
        r_lam = r0 + lam1*r1 + lam2*r2 + lam3*r3
        dual_result = -0.5*q_lam.T.dot(P_lam).dot(q_lam) + r_lam
        print(dual_result.shape)
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
        x = Variable(n)
        objective = Minimize(sum_entries(square(A*x - b)))
        constraints = [0 <= x, x <= 1]
        p = Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        result = p.solve()
        # The optimal value for x is stored in x.value.
        print(x.value)
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print(constraints[0].dual_value)

        ####################################################

        # Scalar variable.
        a = Variable()

        # Column vector variable of length 5.
        x = Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = Variable(4, 7)

        ####################################################

        # Positive scalar parameter.
        m = Parameter(sign="positive")

        # Column vector parameter with unknown sign (by default).
        c = Parameter(5)

        # Matrix parameter with negative entries.
        G = Parameter(4, 7, sign="negative")

        # Assigns a constant value to G.
        G.value = -numpy.ones((4, 7))

        # Raises an error for assigning a value with invalid sign.
        with self.assertRaises(Exception) as cm:
            G.value = numpy.ones((4,7))
        self.assertEqual(str(cm.exception), "Invalid sign for Parameter value.")

        ####################################################
        a = Variable()
        x = Variable(5)

        # expr is an Expression object after each assignment.
        expr = 2*x
        expr = expr - a
        expr = sum_entries(expr) + norm(x, 2)

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
        objective = Minimize(sum_entries(square(A*x - b)) + gamma*norm(x, 1))
        p = Problem(objective)

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
        gamma = Parameter(sign="positive")
        gamma.value = 1
        x = Variable(n)

        # Constants:
        # mu is the vector of expected returns.
        # sigma is the covariance matrix.
        # gamma is a Parameter that trades off risk and return.

        # Variables:
        # x is a vector of stock holdings as fractions of total assets.

        expected_return = mu*x
        risk = quad_form(x, sigma)

        objective = Maximize(expected_return - gamma*risk)
        p = Problem(objective, [sum_entries(x) == 1])
        result = p.solve()

        # The optimal expected return.
        print(expected_return.value)

        # The optimal risk.
        print(risk.value)

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
        gamma = Parameter(sign="positive")
        gamma.value = 0.1
        # 'a' is a variable constrained to have at most 6 non-zero entries.
        a = Variable(n)#mi.SparseVar(n, nonzeros=6)
        b = Variable()

        slack = [pos(1 - label*(sample.T*a - b)) for (label, sample) in data]
        objective = Minimize(norm(a, 2) + gamma*sum(slack))
        p = Problem(objective)
        # Extensions can attach new solve methods to the CVXPY Problem class.
        #p.solve(method="admm")
        p.solve()

        # Count misclassifications.
        errors = 0
        for label, sample in data:
            if label*(sample.T*a - b).value < 0:
                errors += 1

        print("%s misclassifications" % errors)
        print(a.value)
        print(b.value)

    def test_advanded(self):
        """Code from the advanced tutorial.
        """
        # Solving a problem with different solvers.
        x = Variable(2)
        obj = Minimize(x[0] + norm(x, 1))
        constraints = [x >= 2]
        prob = Problem(obj, constraints)

        # Solve with ECOS.
        prob.solve(solver=ECOS)
        print("optimal value with ECOS:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with ECOS_BB.
        prob.solve(solver=ECOS_BB)
        print("optimal value with ECOS_BB:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with CVXOPT.
        prob.solve(solver=CVXOPT)
        print("optimal value with CVXOPT:", prob.value)
        self.assertAlmostEqual(prob.value, 6)

        # Solve with SCS.
        prob.solve(solver=SCS)
        print("optimal value with SCS:", prob.value)
        self.assertAlmostEqual(prob.value, 6, places=2)

        if GLPK in installed_solvers():
            # Solve with GLPK.
            prob.solve(solver=GLPK)
            print("optimal value with GLPK:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

            # Solve with GLPK_MI.
            prob.solve(solver=GLPK_MI)
            print("optimal value with GLPK_MI:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        if GUROBI in installed_solvers():
            # Solve with Gurobi.
            prob.solve(solver=GUROBI)
            print("optimal value with GUROBI:", prob.value)
            self.assertAlmostEqual(prob.value, 6)

        print(installed_solvers())

    def test_log_det(self):
        # Generate data
        x = np.matrix("0.55  0.0;"
                      "0.25  0.35;"
                      "-0.2   0.2;"
                      "-0.25 -0.1;"
                      "-0.0  -0.3;"
                      "0.4  -0.2").T
        (n, m) = x.shape

        # Create and solve the model
        A = Variable(n, n);
        b = Variable(n);
        obj = Maximize( log_det(A) )
        constraints = []
        for i in range(m):
            constraints.append( norm(A*x[:, i] + b) <= 1 )
        p = Problem(obj, constraints)
        result = p.solve()
        self.assertAlmostEqual(result, 1.9746, places=4)

    def test_portfolio_problem(self):
        """Test portfolio problem that caused dcp_attr errors.
        """
        import numpy as np
        import scipy.sparse as sp
        np.random.seed(5)
        n = 100#10000
        m = 10#100
        pbar = (np.ones((n, 1)) * .03 +
                np.matrix(np.append(np.random.rand(n - 1, 1), 0)).T * .12)

        F = sp.rand(m, n, density=0.01)
        F.data = np.ones(len(F.data))
        D = sp.eye(n).tocoo()
        D.data = np.random.randn(len(D.data))**2
        Z = np.random.randn(m, 1)
        Z = Z.dot(Z.T)

        x = Variable(n)
        y = x.__rmul__(F)
        mu = 1
        ret = pbar.T * x
        # DCP attr causes error because not all the curvature
        # matrices are reduced to constants when an atom
        # is scalar.
        risk = square(norm(x.__rmul__(D))) + square(Z*y)

    def test_intro(self):
        """Test examples from cvxpy.org introduction.
        """
        import numpy

        # Problem data.
        m = 30
        n = 20
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m, 1)

        # Construct the problem.
        x = Variable(n)
        objective = Minimize(sum_squares(A*x - b))
        constraints = [0 <= x, x <= 1]
        prob = Problem(objective, constraints)

        # The optimal objective is returned by p.solve().
        result = prob.solve()
        # The optimal value for x is stored in x.value.
        print(x.value)
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        print(constraints[0].dual_value)

        ########################################

        # Create two scalar variables.
        x = Variable()
        y = Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = Minimize(square(x - y))

        # Form and solve problem.
        prob = Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

        ########################################

        import cvxpy as cvx

        # Create two scalar variables.
        x = cvx.Variable()
        y = cvx.Variable()

        # Create two constraints.
        constraints = [x + y == 1,
                       x - y >= 1]

        # Form objective.
        obj = cvx.Minimize(cvx.square(x - y))

        # Form and solve problem.
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("optimal var", x.value, y.value)

        self.assertEqual(prob.status, OPTIMAL)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 0)

        ########################################

        # Replace the objective.
        prob.objective = Maximize(x + y)
        print("optimal value", prob.solve())

        self.assertAlmostEqual(prob.value, 1.0)

        # Replace the constraint (x + y == 1).
        prob.constraints[0] = (x + y <= 3)
        print("optimal value", prob.solve())

        self.assertAlmostEqual(prob.value, 3.0)

        ########################################

        x = Variable()

        # An infeasible problem.
        prob = Problem(Minimize(x), [x >= 1, x <= 0])
        prob.solve()
        print("status:", prob.status)
        print("optimal value", prob.value)

        self.assertEquals(prob.status, INFEASIBLE)
        self.assertAlmostEqual(prob.value, np.inf)

        # An unbounded problem.
        prob = Problem(Minimize(x))
        prob.solve()
        print("status:", prob.status)
        print("optimal value", prob.value)

        self.assertEquals(prob.status, UNBOUNDED)
        self.assertAlmostEqual(prob.value, -np.inf)

        ########################################

        # A scalar variable.
        a = Variable()

        # Column vector variable of length 5.
        x = Variable(5)

        # Matrix variable with 4 rows and 7 columns.
        A = Variable(4, 7)

        ########################################
        import numpy

        # Problem data.
        m = 10
        n = 5
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m, 1)

        # Construct the problem.
        x = Variable(n)
        objective = Minimize(sum_entries(square(A*x - b)))
        constraints = [0 <= x, x <= 1]
        prob = Problem(objective, constraints)

        print("Optimal value", prob.solve())
        print("Optimal var")
        print(x.value) # A numpy matrix.

        self.assertAlmostEqual(prob.value, 4.14133859146)

        ########################################
        # Positive scalar parameter.
        m = Parameter(sign="positive")

        # Column vector parameter with unknown sign (by default).
        c = Parameter(5)

        # Matrix parameter with negative entries.
        G = Parameter(4, 7, sign="negative")

        # Assigns a constant value to G.
        G.value = -numpy.ones((4, 7))
        ########################################

        # Create parameter, then assign value.
        rho = Parameter(sign="positive")
        rho.value = 2

        # Initialize parameter with a value.
        rho = Parameter(sign="positive", value=2)

        ########################################

        import numpy

        # Problem data.
        n = 15
        m = 10
        numpy.random.seed(1)
        A = numpy.random.randn(n, m)
        b = numpy.random.randn(n, 1)
        # gamma must be positive due to DCP rules.
        gamma = Parameter(sign="positive")

        # Construct the problem.
        x = Variable(m)
        error = sum_squares(A*x - b)
        obj = Minimize(error + gamma*norm(x, 1))
        prob = Problem(obj)

        # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
        sq_penalty = []
        l1_penalty = []
        x_values = []
        gamma_vals = numpy.logspace(-4, 6)
        for val in gamma_vals:
            gamma.value = val
            prob.solve()
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            sq_penalty.append(error.value)
            l1_penalty.append(norm(x, 1).value)
            x_values.append(x.value)

        ########################################
        import numpy

        X = Variable(5, 4)
        A = numpy.ones((3, 5))

        # Use expr.size to get the dimensions.
        print("dimensions of X:", X.size)
        print("dimensions of sum_entries(X):", sum_entries(X).size)
        print("dimensions of A*X:", (A*X).size)

        # ValueError raised for invalid dimensions.
        try:
            A + X
        except ValueError as e:
            print(e)

    def test_inpainting(self):
        """Test image in-painting.
        """
        import numpy as np
        np.random.seed(1)
        rows, cols = 100, 100
        # Load the images.
        # Convert to arrays.
        Uorig = np.random.randint(0, 255, size=(rows, cols))

        rows, cols = Uorig.shape
        # Known is 1 if the pixel is known,
        # 0 if the pixel was corrupted.
        Known = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.random() > 0.7:
                    Known[i, j] = 1
        Ucorr = Known*Uorig
        # Recover the original image using total variation in-painting.
        U = Variable(rows, cols)
        obj = Minimize(tv(U))
        constraints = [mul_elemwise(Known, U) == mul_elemwise(Known, Ucorr)]
        prob = Problem(obj, constraints)
        prob.solve(solver=SCS)

    def test_advanced(self):
        """Test code from the advanced section of the tutorial.
        """
        x = Variable()
        prob = Problem(Minimize(square(x)), [x == 2])
        # Get ECOS arguments.
        data = prob.get_problem_data(ECOS)

        # Get ECOS_BB arguments.
        data = prob.get_problem_data(ECOS_BB)

        # Get CVXOPT arguments.
        data = prob.get_problem_data(CVXOPT)

        # Get SCS arguments.
        data = prob.get_problem_data(SCS)

        import ecos
        # Get ECOS arguments.
        data = prob.get_problem_data(ECOS)
        # Call ECOS solver.
        solver_output = ecos.solve(data["c"], data["G"], data["h"],
                                   data["dims"], data["A"], data["b"])
        # Unpack raw solver output.
        prob.unpack_results(ECOS, solver_output)

    def test_log_sum_exp(self):
        """Test log_sum_exp function that failed in Github issue.
        """
        import cvxpy as cp
        import numpy as np
        np.random.seed(1)
        m = 5
        n = 2
        X = np.matrix(np.ones((m,n)))
        w = cp.Variable(n)

        expr2 = [cp.log_sum_exp(cp.vstack(0, X[i,:]*w)) for i in range(m)]
        expr3 = sum(expr2)
        obj = cp.Minimize(expr3)
        p = cp.Problem(obj)
        p.solve(solver=SCS, max_iters=1)

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
    #     constraints = [sum_entries(x) == 1, x >= 0]
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