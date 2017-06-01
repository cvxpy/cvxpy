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
from scipy.linalg import lstsq

from cvxpy import Minimize, Problem
from cvxpy.atoms import QuadForm, abs, power, quad_over_lin, sum_entries, sum_squares
from cvxpy.expressions.variables import Variable
from cvxpy.solver_interface.qp_solvers.qp_solver import QpSolver
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

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

        self.solvers = ['GUROBI', 'MOSEK', 'qpOASES', 'OSQP']

    def test_all_solvers(self):
        for solver in self.solvers:
            self.quad_over_lin(solver)
            self.power(solver)
            self.power_matrix(solver)
            self.square_affine(solver)
            self.quad_form(solver)
            self.affine_problem(solver)

    def quad_over_lin(self, solver):
        p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x-1), 1)), [self.x <= -1])
        s = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual(numpy.array([-1., -1.]), s.primal_vars[var.id])
        for con in p.constraints:
            self.assertItemsAlmostEqual(numpy.array([2., 2.]), s.dual_vars[con.id])

    def power(self, solver):
        p = Problem(Minimize(sum_entries(power(self.x, 2))), [])
        s = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], s.primal_vars[var.id])

    def power_matrix(self, solver):
        p = Problem(Minimize(sum_entries(power(self.A - 3., 2))), [])
        s = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual([3., 3., 3., 3.], s.primal_vars[var.id])

    def square_affine(self, solver):
        A = numpy.random.randn(10, 2)
        b = numpy.random.randn(10, 1)
        p = Problem(Minimize(sum_squares(A*self.x - b)))
        s = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), s.primal_vars[var.id], places=1)

    def quad_form(self, solver):
        numpy.random.seed(0)
        A = numpy.random.randn(5, 5)
        z = numpy.random.randn(5, 1)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        qp_solution = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual(z, qp_solution.primal_vars[var.id])

    def affine_problem(self, solver):
        A = numpy.random.randn(5, 2)
        A = numpy.maximum(A, 0)
        b = numpy.random.randn(5, 1)
        b = numpy.maximum(b, 0)
        p = Problem(Minimize(sum_entries(self.x)), [self.x >= 0, A*self.x <= b])
        s = QpSolver(solver).solve(p, False, False, {})
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], s.primal_vars[var.id])
