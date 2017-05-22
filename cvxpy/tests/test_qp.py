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

from cvxpy import Maximize, Minimize, Problem
from cvxpy.atoms import *
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Bool, Semidef, Symmetric, Variable
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.qp2quad_form.qp2quad_form import Qp2QuadForm
from cvxpy.solver_interface.qp_solvers.qp_solver import QpSolver
from cvxpy.tests.base_test import BaseTest
from scipy.linalg import lstsq

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

    def solve_QP(self, problem, solver):
        self.assertTrue(Qp2QuadForm().accepts(problem))
        canon_p, canon_inverse = Qp2QuadForm().apply(problem)
        self.assertTrue(QpMatrixStuffing().accepts(canon_p))
        stuffed_p, stuffed_inverse = QpMatrixStuffing().apply(canon_p)
        qp_solution = QpSolver(solver).solve(stuffed_p, False, False, {})
        stuffed_solution = QpMatrixStuffing().invert(qp_solution, stuffed_inverse)
        return Qp2QuadForm().invert(stuffed_solution, canon_inverse)

    def test_quad_over_lin(self):
        p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x-1), 1)), [self.x <= -1])
        s = self.solve_QP(p, 'GUROBI')
        for var in p.variables():
            self.assertItemsAlmostEqual(numpy.array([-1., -1.]), s.primal_vars[var.id])
        for con in p.constraints:
            self.assertItemsAlmostEqual(numpy.array([2., 2.]), s.dual_vars[con.id])

    def test_power(self):
        p = Problem(Minimize(sum_entries(power(self.x, 2))), [])
        s = self.solve_QP(p, 'GUROBI')
        for var in p.variables():
            self.assertItemsAlmostEqual([0., 0.], s.primal_vars[var.id])

    def test_power_matrix(self):
        p = Problem(Minimize(sum_entries(power(self.A - 3., 2))), [])
        s = self.solve_QP(p, 'GUROBI')
        for var in p.variables():
            self.assertItemsAlmostEqual([3., 3., 3., 3.], s.primal_vars[var.id])

    def test_square_affine(self):
        A = numpy.random.randn(10,2)
        b = numpy.random.randn(10,1)
        p = Problem(Minimize(sum_squares(A*self.x - b)))
        s = self.solve_QP(p, 'GUROBI')
        for var in p.variables():
            self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), s.primal_vars[var.id], places=1)

    def test_quad_form(self):
        numpy.random.seed(0)
        A = numpy.random.randn(5,5)
        z = numpy.random.randn(5,1)
        P = A.T.dot(A)
        q = -2*P.dot(z)
        p = Problem(Minimize(QuadForm(self.w, P) + q.T*self.w))
        s = self.solve_QP(p, 'GUROBI')
        for var in p.variables():
            self.assertItemsAlmostEqual(z, s.primal_vars[var.id])

    

