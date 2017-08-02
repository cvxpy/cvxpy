"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

from cvxpy import Minimize, Problem
from cvxpy.atoms import (QuadForm, abs, power, quad_over_lin, sum, sum_squares, norm,
                         matrix_frac)
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.tests.base_test import BaseTest


class TestQp(BaseTest):
    """ Unit tests for the domain module. """

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

        self.xs = Variable(800, name='xs')
        self.xsr = Variable(500, name='xsr')
        self.xef = Variable(80, name='xef')

        self.solvers = ['GUROBI']

    def solve_QP(self, problem, solver):
        self.assertTrue(Qp2SymbolicQp().accepts(problem))
        canon_p, canon_inverse = Qp2SymbolicQp().apply(problem)
        self.assertTrue(QpMatrixStuffing().accepts(canon_p))
        stuffed_p, stuffed_inverse = QpMatrixStuffing().apply(canon_p)
        qp_solution = QpSolver(solver).solve(stuffed_p, False, False, {})
        stuffed_solution = QpMatrixStuffing().invert(qp_solution, stuffed_inverse)
        return Qp2SymbolicQp().invert(stuffed_solution, canon_inverse)

    def test_all_solvers(self):
        for solver in self.solvers:
#            self.quad_over_lin(solver)
#            self.power(solver)
#            self.power_matrix(solver)
#            self.square_affine(solver)
#            self.quad_form(solver)
#            self.affine_problem(solver)
            # Do we need the following functionality?
            # self.norm_2(solver)
            # self.mat_norm_2(solver)
#            self.quad_form_coeff(solver)
#            self.quad_form_bound(solver)
#            self.regression_1(solver)
            self.regression_2(solver)
            # slow tests:
            # self.control(solver)
            # self.sparse_system(solver)
#            self.smooth_ridge(solver)
#            self.equivalent_forms_1(solver)
#            self.equivalent_forms_2(solver)
#            self.equivalent_forms_3(solver)

    def quad_over_lin(self, solver):
        p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x-1), 1)), [self.x <= -1])
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(numpy.array([-1., -1.]), s.primal_vars[var.id])
        for con in p.constraints:
            self.assertItemsAlmostEqual(numpy.array([2., 2.]), s.dual_vars[con.id])

    def power(self, solver):
        p = Problem(Minimize(sum(power(self.x, 2))), [])
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], s.primal_vars[var.id])

    def power_matrix(self, solver):
        p = Problem(Minimize(sum(power(self.A - 3., 2))), [])
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([3., 3., 3., 3.], s.primal_vars[var.id])

    def square_affine(self, solver):
        A = numpy.random.randn(10, 2)
        b = numpy.random.randn(10, 1)
        p = Problem(Minimize(sum_squares(A*self.x - b)))
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), s.primal_vars[var.id], places=1)

    def quad_form(self, solver):
        numpy.random.seed(0)
        A = numpy.random.randn(5, 5)
        z = numpy.random.randn(5, 1)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        qp_solution = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, qp_solution.primal_vars[var.id])

    def affine_problem(self, solver):
        A = numpy.random.randn(5, 2)
        A = numpy.maximum(A, 0)
        b = numpy.random.randn(5, 1)
        b = numpy.maximum(b, 0)
        p = Problem(Minimize(sum(self.x)), [self.x >= 0, A*self.x <= b])
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], s.primal_vars[var.id])

    def norm_2(self, solver):
        A = numpy.random.randn(10, 5)
        b = numpy.random.randn(10, 1)
        p = Problem(Minimize(norm(A*self.w - b, 2)))
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), s.primal_vars[var.id], places=1)

    def mat_norm_2(self, solver):
        A = numpy.random.randn(5, 3)
        B = numpy.random.randn(5, 2)
        p = Problem(Minimize(norm(A*self.C - B, 2)))
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, B)[0], s.primal_vars[var.id], places=1)

    def quad_form_coeff(self, solver):
        numpy.random.seed(0)
        A = numpy.random.randn(5, 5)
        z = numpy.random.randn(5, 1)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        qp_solution = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(z, qp_solution.primal_vars[var.id])

    def quad_form_bound(self, solver):
        P = numpy.matrix([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
        q = numpy.matrix([[-22], [-14.5], [13]])
        r = 1
        y_star = numpy.matrix([[1], [0.5], [-1]])
        p = Problem(Minimize(0.5*QuadForm(self.y, P) + q.T*self.y + r), [self.y >= -1, self.y <= 1])
        s = self.solve_QP(p, solver)
        for var in p.variables():
            self.assertItemsAlmostEqual(y_star, s.primal_vars[var.id])

    def regression_1(self, solver):
        numpy.random.seed(1)
        # Number of examples to use
        n = 100
        # Specify the true value of the variable
        true_coeffs = numpy.matrix('2; -2; 0.5')
        # Generate data
        x_data = numpy.random.rand(n, 1) * 5
        x_data = numpy.asmatrix(x_data)
        x_data_expanded = numpy.hstack([numpy.power(x_data, i) for i in range(1, 4)])
        x_data_expanded = numpy.asmatrix(x_data_expanded)
        y_data = x_data_expanded * true_coeffs + 0.5 * numpy.random.rand(n, 1)
        y_data = numpy.asmatrix(y_data)

        line = self.offset + x_data * self.slope
        residuals = line - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        s = self.solve_QP(p, solver)
        self.assertAlmostEqual(1171.60037715, s.opt_val)

    def regression_2(self, solver):
        numpy.random.seed(1)
        # Number of examples to use
        n = 100
        # Specify the true value of the variable
        true_coeffs = numpy.matrix('2; -2; 0.5')
        # Generate data
        x_data = numpy.random.rand(n, 1) * 5
        x_data = numpy.asmatrix(x_data)
        x_data_expanded = numpy.hstack([numpy.power(x_data, i) for i in range(1, 4)])
        x_data_expanded = numpy.asmatrix(x_data_expanded)
        y_data = x_data_expanded * true_coeffs + 0.5 * numpy.random.rand(n, 1)
        y_data = numpy.asmatrix(y_data)

        quadratic = self.offset + x_data*self.slope + self.quadratic_coeff*numpy.power(x_data, 2)
        residuals = quadratic - y_data
        fit_error = sum_squares(residuals)
        p = Problem(Minimize(fit_error), [])
        s = self.solve_QP(p, solver)
        self.assertAlmostEqual(139.225660756, s.opt_val)

    def control(self, solver):
        # Some constraints on our motion
        # The object should start from the origin, and end at rest
        initial_velocity = numpy.matrix('-20; 100')
        final_position = numpy.matrix('100; 100')
        T = 100  # The number of timesteps
        h = 0.1  # The time between time intervals
        mass = 1  # Mass of object
        drag = 0.1  # Drag on object
        g = numpy.matrix('0; -9.8')  # Gravity on object
        # Create a problem instance
        constraints = []
        # Add constraints on our variables
        for i in range(T - 1):
            constraints += [self.position[:, i + 1] == self.position[:, i] + h*self.velocity[:, i]]
            acceleration = self.force[:, i]/mass + g - drag * self.velocity[:, i]
            constraints += [self.velocity[:, i + 1] == self.velocity[:, i] + h * acceleration]
        # Add position constraints
        constraints += [self.position[:, 0] == 0]
        constraints += [self.position[:, -1] == final_position]
        # Add velocity constraints
        constraints += [self.velocity[:, 0] == initial_velocity]
        constraints += [self.velocity[:, -1] == 0]
        # Solve the problem
        p = Problem(Minimize(sum_squares(self.force)), constraints)
        s = self.solve_QP(p, solver)
        self.assertAlmostEqual(17850.0, s.opt_val)

    def sparse_system(self, solver):
        m = 1000
        n = 800
        numpy.random.seed(1)
        density = 0.2
        A = sp.rand(m, n, density)
        b = numpy.random.randn(m, 1)

        p = Problem(Minimize(sum_squares(A*self.xs - b)), [self.xs == 0])
        s = self.solve_QP(p, solver)
        self.assertAlmostEqual(b.T.dot(b), s.opt_val)

    def smooth_ridge(self, solver):
        numpy.random.seed(1)
        n = 500
        k = 50
        eta = 1

        A = numpy.ones((k, n))
        b = numpy.ones((k, 1))
        obj = sum_squares(A*self.xsr - b) + eta*sum_squares(self.xsr[:-1]-self.xsr[1:])
        p = Problem(Minimize(obj), [])
        s = self.solve_QP(p, solver)
        self.assertAlmostEqual(0, s.opt_val)

    def equivalent_forms_1(self, solver):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m, 1)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r, 1)

        obj1 = sum((A*self.xef - b) ** 2)
        cons = [G*self.xef == h]

        p1 = Problem(Minimize(obj1), cons)
        s = self.solve_QP(p1, solver)
        self.assertAlmostEqual(s.opt_val, 681.119420108)

    def equivalent_forms_2(self, solver):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m, 1)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r, 1)

        # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
        P = numpy.dot(A.T, A)
        q = -2*numpy.dot(A.T, b)
        r = numpy.dot(b.T, b)

        obj2 = QuadForm(self.xef, P)+q.T*self.xef+r
        cons = [G*self.xef == h]

        p2 = Problem(Minimize(obj2), cons)
        s = self.solve_QP(p2, solver)
        self.assertAlmostEqual(s.opt_val, 681.119420108)

    def equivalent_forms_3(self, solver):
        m = 100
        n = 80
        r = 70
        numpy.random.seed(1)
        A = numpy.random.randn(m, n)
        b = numpy.random.randn(m, 1)
        G = numpy.random.randn(r, n)
        h = numpy.random.randn(r, 1)

        # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
        P = numpy.dot(A.T, A)
        q = -2*numpy.dot(A.T, b)
        r = numpy.dot(b.T, b)
        Pinv = numpy.linalg.inv(P)

        obj3 = matrix_frac(self.xef, Pinv)+q.T*self.xef+r
        cons = [G*self.xef == h]

        p3 = Problem(Minimize(obj3), cons)
        s = self.solve_QP(p3, solver)
        self.assertAlmostEqual(s.opt_val, 681.119420108)
