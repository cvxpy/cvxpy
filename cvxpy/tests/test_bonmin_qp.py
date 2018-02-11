"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren,
          2018 Sascha-Dominic Schnug

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

import numpy
import scipy.sparse as sp
from scipy.linalg import lstsq

import cvxpy.settings as s
from cvxpy import Minimize, Problem, Parameter, Maximize
from cvxpy.atoms import (QuadForm, abs, power,
                         quad_over_lin, sum, sum_squares,
                         norm,
                         matrix_frac,
                         hstack)
from cvxpy.reductions.solvers.defines \
    import SOLVER_MAP_QP, QP_SOLVERS, INSTALLED_SOLVERS
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.tests.base_test import BaseTest


class TestQp(BaseTest):
    """ Unit tests for the domain module.

        This test-module heavily borrows from test_qp.py and was modified for
        BONMIN_QP: no duals, no warm-start!

        Two configs are tested: hessian=exact vs. hessian='limited-memory'.

        For continuous QPs, only Ipopt will be used.
    """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')
        self.w = Variable(5, name='w')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

        self.slope = Variable(1, name='slope')
        self.offset = Variable(1, name='offset')
        self.quadratic_coeff = Variable(1, name='quadratic_coeff')

        T = 100
        self.position = Variable((2, T), name='position')
        self.velocity = Variable((2, T), name='velocity')
        self.force = Variable((2, T - 1), name='force')

        self.xs = Variable(80, name='xs')
        self.xsr = Variable(200, name='xsr')
        self.xef = Variable(80, name='xef')

    def solve_QP(self, problem, solver_name, hessian_approximation):
        return problem.solve(solver=solver_name, verbose=True,
                             hessian_approximation=hessian_approximation)

    def test_all_problems(self):
        solver = 'BONMIN_QP'

        # 2 configs: exact vs. approximate-hessian
        configs = [('hessian_approximation', False),
                   ('hessian_approximation', True)]

        for config in configs:
            self.quad_over_lin(solver, hess_approx=config[1])
            self.power(solver, hess_approx=config[1])
            self.power_matrix(solver, hess_approx=config[1])
            self.square_affine(solver, hess_approx=config[1])
            self.quad_form(solver, hess_approx=config[1])
            self.affine_problem(solver, hess_approx=config[1])
            self.maximize_problem(solver, hess_approx=config[1])

            # Do we need the following functionality?
            # self.norm_2(solver)
            # self.mat_norm_2(solver)

            self.quad_form_coeff(solver, hess_approx=config[1])
            self.quad_form_bound(solver, hess_approx=config[1])
            self.regression_1(solver, hess_approx=config[1])
            self.regression_2(solver, hess_approx=config[1])

            # slow tests:
            self.control(solver, hess_approx=config[1])
            self.sparse_system(solver, hess_approx=config[1])
            self.smooth_ridge(solver, hess_approx=config[1])
            self.equivalent_forms_1(solver, hess_approx=config[1])
            self.equivalent_forms_2(solver, hess_approx=config[1])
            self.equivalent_forms_3(solver, hess_approx=config[1])

    def quad_over_lin(self, solver, hess_approx):
        p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x-1), 1)),
                    [self.x <= -1])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(numpy.array([-1., -1.]),
                                        var.value)
        # No duals available!
        # for con in p.constraints:
        #     self.assertItemsAlmostEqual(numpy.array([2., 2.]),
        #                                 con.dual_value)

    def power(self, solver, hess_approx):
        p = Problem(Minimize(sum(power(self.x, 2))), [])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], var.value)

    def power_matrix(self, solver, hess_approx):
        p = Problem(Minimize(sum(power(self.A - 3., 2))), [])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual([3., 3., 3., 3.],
                                        var.value)

    def square_affine(self, solver, hess_approx):
        A = numpy.random.randn(10, 2)
        b = numpy.random.randn(10)
        p = Problem(Minimize(sum_squares(A*self.x - b)))
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(),
                                        var.value,
                                        places=1)

    def quad_form(self, solver, hess_approx):
        numpy.random.seed(0)
        A = numpy.random.randn(5, 5)
        z = numpy.random.randn(5)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        qp_solution = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, var.value)

    def affine_problem(self, solver, hess_approx):
        A = numpy.random.randn(5, 2)
        A = numpy.maximum(A, 0)
        b = numpy.random.randn(5)
        b = numpy.maximum(b, 0)
        p = Problem(Minimize(sum(self.x)), [self.x >= 0, A*self.x <= b])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], var.value, places=4)

    def maximize_problem(self, solver, hess_approx):
        A = numpy.random.randn(5, 2)
        A = numpy.maximum(A, 0)
        b = numpy.random.randn(5)
        b = numpy.maximum(b, 0)
        p = Problem(Maximize(-sum(self.x)), [self.x >= 0, A*self.x <= b])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], var.value, places=4)

    def norm_2(self, solver, hess_approx):
        A = numpy.random.randn(10, 5)
        b = numpy.random.randn(10)
        p = Problem(Minimize(norm(A*self.w - b, 2)))
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), var.value,
                                        places=1)

    def mat_norm_2(self, solver, hess_approx):
        A = numpy.random.randn(5, 3)
        B = numpy.random.randn(5, 2)
        p = Problem(Minimize(norm(A*self.C - B, 2)))
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, B)[0],
                                        s.primal_vars[var.id], places=1)

    def quad_form_coeff(self, solver, hess_approx):
        numpy.random.seed(0)
        A = numpy.random.randn(5, 5)
        z = numpy.random.randn(5)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        qp_solution = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, var.value)

    def quad_form_bound(self, solver, hess_approx):
        P = numpy.matrix([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
        q = numpy.matrix([[-22], [-14.5], [13]])
        r = 1
        y_star = numpy.matrix([[1], [0.5], [-1]])
        p = Problem(Minimize(0.5*QuadForm(self.y, P) + q.T*self.y + r),
                    [self.y >= -1, self.y <= 1])
        s = self.solve_QP(p, solver, hess_approx)
        for var in p.variables():
            self.assertItemsAlmostEqual(y_star, var.value)

    def regression_1(self, solver, hess_approx):
        numpy.random.seed(1)
        # Number of examples to use
        n = 100
        # Specify the true value of the variable
        true_coeffs = numpy.matrix('2; -2; 0.5')
        # Generate data
        x_data = numpy.random.rand(n) * 5
        x_data = numpy.asmatrix(x_data)
        x_data_expanded = numpy.vstack([numpy.power(x_data, i)
                                        for i in range(1, 4)])
        x_data_expanded = numpy.asmatrix(x_data_expanded)
        y_data = x_data_expanded.T * true_coeffs + 0.5 * numpy.random.rand(n,1)
        y_data = numpy.asmatrix(y_data)

        line = self.offset + x_data * self.slope
        residuals = line.T - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        s = self.solve_QP(p, solver, hess_approx)
        self.assertAlmostEqual(1171.60037715, p.value)

    def regression_2(self, solver, hess_approx):
        numpy.random.seed(1)
        # Number of examples to use
        n = 100
        # Specify the true value of the variable
        true_coeffs = numpy.matrix('2; -2; 0.5')
        # Generate data
        x_data = numpy.random.rand(n) * 5
        x_data = numpy.asmatrix(x_data)
        x_data_expanded = numpy.vstack([numpy.power(x_data, i)
                                        for i in range(1, 4)])
        x_data_expanded = numpy.asmatrix(x_data_expanded)
        y_data = x_data_expanded.T * true_coeffs + 0.5 * numpy.random.rand(n, 1)
        y_data = numpy.asmatrix(y_data)

        quadratic = self.offset + x_data*self.slope + \
            self.quadratic_coeff*numpy.power(x_data, 2)
        residuals = quadratic.T - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        s = self.solve_QP(p, solver, hess_approx)

        self.assertAlmostEqual(139.225660756, p.value)

    def control(self, solver, hess_approx):
        # Some constraints on our motion
        # The object should start from the origin, and end at rest
        initial_velocity = numpy.array([-20, 100])
        final_position = numpy.array([100, 100])
        T = 100  # The number of timesteps
        h = 0.1  # The time between time intervals
        mass = 1  # Mass of object
        drag = 0.1  # Drag on object
        g = numpy.array([0, -9.8])  # Gravity on object
        # Create a problem instance
        constraints = []
        # Add constraints on our variables
        for i in range(T - 1):
            constraints += [self.position[:, i + 1] == self.position[:, i] +
                            h*self.velocity[:, i]]
            acceleration = self.force[:, i]/mass + g - \
                drag * self.velocity[:, i]
            constraints += [self.velocity[:, i + 1] == self.velocity[:, i] +
                            h * acceleration]

        # Add position constraints
        constraints += [self.position[:, 0] == 0]
        constraints += [self.position[:, -1] == final_position]
        # Add velocity constraints
        constraints += [self.velocity[:, 0] == initial_velocity]
        constraints += [self.velocity[:, -1] == 0]
        # Solve the problem
        p = Problem(Minimize(.01 * sum_squares(self.force)), constraints)
        s = self.solve_QP(p, solver, hess_approx)
        self.assertAlmostEqual(178.500, p.value, places=1)

    def sparse_system(self, solver, hess_approx):
        m = 100
        n = 80
        numpy.random.seed(1)
        density = 0.4
        A = sp.rand(m, n, density)
        b = numpy.random.randn(m)

        p = Problem(Minimize(sum_squares(A*self.xs - b)), [self.xs == 0])
        s = self.solve_QP(p, solver, hess_approx)
        self.assertAlmostEqual(b.T.dot(b), p.value)

    def smooth_ridge(self, solver, hess_approx):
        numpy.random.seed(1)
        n = 200
        k = 50
        eta = 1

        A = numpy.ones((k, n))
        b = numpy.ones((k))
        obj = sum_squares(A*self.xsr - b) + \
            eta*sum_squares(self.xsr[:-1]-self.xsr[1:])
        p = Problem(Minimize(obj), [])
        s = self.solve_QP(p, solver, hess_approx)
        self.assertAlmostEqual(0, p.value)

    def equivalent_forms_1(self, solver, hess_approx):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r)

        obj1 = .1 * sum((A*self.xef - b) ** 2)
        cons = [G*self.xef == h]

        p1 = Problem(Minimize(obj1), cons)
        s = self.solve_QP(p1, solver, hess_approx)
        self.assertAlmostEqual(p1.value, 68.1119420108)

    def equivalent_forms_2(self, solver, hess_approx):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r)

        # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
        P = numpy.dot(A.T, A)
        q = -2*numpy.dot(A.T, b)
        r = numpy.dot(b.T, b)

        obj2 = .1*(QuadForm(self.xef, P)+q.T*self.xef+r)
        cons = [G*self.xef == h]

        p2 = Problem(Minimize(obj2), cons)
        s = self.solve_QP(p2, solver, hess_approx)
        self.assertAlmostEqual(p2.value, 68.1119420108)

    def equivalent_forms_3(self, solver, hess_approx):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r)

        # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
        P = numpy.dot(A.T, A)
        q = -2*numpy.dot(A.T, b)
        r = numpy.dot(b.T, b)
        Pinv = numpy.linalg.inv(P)

        obj3 = .1 * (matrix_frac(self.xef, Pinv)+q.T*self.xef+r)
        cons = [G*self.xef == h]

        p3 = Problem(Minimize(obj3), cons)
        s = self.solve_QP(p3, solver, hess_approx)
        self.assertAlmostEqual(p3.value, 68.1119420108)


class TestMiQp(BaseTest):
    """ Unit tests for the domain module.

        Two configs are tested: hessian=exact vs. hessian='limited-memory'.
    """

    def test_0(self):
        """ Small MIQP example from OPTI:
            https://www.inverseproblem.co.nz/OPTI/index.php/Probs/MIQP
        """
        x1 = Variable(integer=True)
        x2 = Variable()
        x = hstack((x1, x2))

        H = numpy.array([[1,-1],[-1,2]])
        f = numpy.array([-2, -6])

        constraints = [  x1 +   x2 <= 2,
                        -x1 + 2*x2 <= 2,
                       2*x1 +   x2 <= 3,
                                 0 <= x]

        p = Problem(Minimize(f*x + 0.5 * QuadForm(x, H)), constraints)

        for algorithm in ['B-Hyb', 'B-BB', 'B-OA', 'B-QG']:
            for hessian_approximation in [False, True]:
                p.solve(solver='BONMIN_QP', verbose=True, algorithm=algorithm,
                        hessian_approximation=hessian_approximation)
                self.assertItemsAlmostEqual(x.value, [1.0, 1.0])
                self.assertAlmostEqual(p.value, -7.5)

    def test_1(self):
        """ Small MIP (in MIQP form) example from
            miqp.m: A Matlab function for solving Mixed Integer Quadratic Programs
            Version 1.02
            User Guide
        """
        x1 = Variable()
        x_2_3_4 = Variable(3, integer=True)
        x = hstack((x1, x_2_3_4))

        Q = numpy.zeros((4,4))
        b = numpy.array([2, -3, -2, -3])
        C = numpy.array([[-1, -1, -1, -1],[10, 5, 3, 4],[-1, 0, 0, 0]])
        d = numpy.array([-2, 10, 0])
        vlb = numpy.array([-1e10, 0, 0, 0])
        vub = numpy.array([1e10, 1, 1, 1])

        constraints = [C*x <= d,
                       vlb <= x,
                       vub >= x]

        p = Problem(Minimize(b*x + 0.5 * QuadForm(x, Q)), constraints)

        for algorithm in ['B-Hyb', 'B-BB', 'B-OA', 'B-QG']:
            for hessian_approximation in [False, True]:
                p.solve(solver='BONMIN_QP', verbose=True, algorithm=algorithm,
                        hessian_approximation=hessian_approximation)
                self.assertItemsAlmostEqual(x.value, [0.0, 1.0, 0.0, 1.0])
                self.assertAlmostEqual(p.value, -6.0)

    def test_2(self):
        """ Small integer least-squares example modified from:
            https://github.com/cvxgrp/cvxpy/blob/master/examples/extensions/integer_ls.py
        """
        x = Variable(3, name='x', boolean=True)
        A = numpy.array([1,2,3,4,5,6,7,8,9]).reshape(3,3)
        z = numpy.array([3, 7, 9])

        p = Problem(Minimize(sum_squares(A*x - z)))

        for algorithm in ['B-Hyb', 'B-BB', 'B-OA', 'B-QG']:
            for hessian_approximation in [False, True]:
                p.solve(solver='BONMIN_QP', verbose=True, algorithm=algorithm,
                        hessian_approximation=hessian_approximation)
                self.assertItemsAlmostEqual(x.value, [0.0, 0.0, 1.0])
                self.assertAlmostEqual(p.value, 1.0)

    def test_3(self):
        """ Small cardinality-constrained nonnegative least-squares example
            modified from:
            http://docs.roguewave.com/imsl/java/7.1/manual/api/com/imsl/math/NonNegativeLeastSquaresEx1.html
        """
        # Basic continuous non-nonneg problem
        x = Variable(3, name='x')
        A = numpy.array([[1, -3, 2],[-3, 10, -5],[2, -5, 6]])
        b = numpy.array([27, -78, 64])

        p = Problem(Minimize(sum_squares(A*x - b)))

        for algorithm in ['B-Hyb', 'B-BB', 'B-OA', 'B-QG']:
            for hessian_approximation in [False, True]:
                p.solve(solver='BONMIN_QP', verbose=True, algorithm=algorithm,
                        hessian_approximation=hessian_approximation)
                self.assertItemsAlmostEqual(x.value, [1.0, -4.0, 7.0])

        # Continuous nonneg problem
        constraints_nonneg = [x >= 0]

        p = Problem(Minimize(sum_squares(A*x - b)), constraints_nonneg)

        for algorithm in ['B-Hyb', 'B-BB', 'B-OA', 'B-QG']:
            for hessian_approximation in [False, True]:
                p.solve(solver='BONMIN_QP', verbose=True, algorithm=algorithm,
                        hessian_approximation=hessian_approximation)
                self.assertItemsAlmostEqual(x.value, [1.84492754e+01, 1.04260926e-08,
                    4.50724637e+00])  # ECOS-BB result

        # Cardinality-constrained nonneg problem
        # Simple linearization following:
        # https://orinanobworld.blogspot.de/2010/10/binary-variables-and-quadratic-terms.html
        upper_bound = 1000.
        card_n = 2
        y = Variable(x.size, name='y', boolean=True)
        z = Variable(x.size, name='z')
        constraints_card = [z <= upper_bound * y,
                            z >= 0,                          # lb=0
                            z <= x,                          # lb=0
                            z >= x - upper_bound*(1-y),
                            sum(y) <= card_n]                # cardinality

        p = Problem(Minimize(sum_squares(A*z - b)),
                constraints_nonneg + constraints_card)

        p.solve(solver='BONMIN_QP', verbose=True)
        # p.solve(solver='BONMIN_QP', algorithm='B-Hyb', verbose=True) # BUG!

        # compared to ECOS_BB (which is of lower-precision)
        self.assertItemsAlmostEqual(y.value,
            [0.9999999999993863, 1.145048366236877e-12, 0.9999999999993624])
        self.assertItemsAlmostEqual(z.value,
            [18.449275056142078, 1.1645657355651684e-10, 4.507246518757189])


class TestStatus(BaseTest):
    """ Basic tests for status in (optimal, infeasible, unbounded)
    """

    def test_0(self):
        # feasible -> optimal
        # TODO BROKEN!
        # -> bonmin-research needed => if z bool: no bounds on z in LP.relax added?
        x = Variable()
        y = Variable(integer=True)
        z = Variable(boolean=True)

        constraints = [0 <= x, x <=5,
                      -3 <= y, y <=2]

        p = Problem(Maximize(x + y + z), constraints)
        p.solve(solver='BONMIN_QP', verbose=True)
        assert p.status == s.OPTIMAL

    def test_1(self):
        # infeasible
        # TODO BROKEN
        # -> bonmin-research needed => if z bool: no bounds on z in LP.relax added?
        x = Variable()
        y = Variable(integer=True)
        z = Variable(boolean=True)

        constraints = [0 <= x, x <=5,
                      -3 <= y, y <=2,
                       x + y + z >= 9]

        p = Problem(Minimize(x + y + z), constraints)
        p.solve(solver='BONMIN_QP', verbose=True)
        assert p.status == s.INFEASIBLE

    def test_2(self):
        # unbounded
        # TODO BROKEN
        x = Variable()
        y = Variable(integer=True)
        z = Variable(boolean=True)

        constraints = [0 <= x, x <=5,
                      -3 <= y,
                       x + y + z >= 9]

        p = Problem(Maximize(x + y + z), constraints)
        p.solve(solver='BONMIN_QP', verbose=True)
        assert p.status == s.UNBOUNDED
