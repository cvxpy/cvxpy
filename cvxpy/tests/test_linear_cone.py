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
from cvxpy.atoms import exp, diag, pnorm
from cvxpy.constraints import SOC, ExpCone
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.tests.base_test import BaseTest


class TestLinearCone(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

        # TODO(akshayka): Why are these solvers commented out? If it is
        # because their interfaces are not yet implemented, then a comment
        # along those lines should exist.
        self.solvers = [ECOS()]#, GUROBI(), MOSEK(), SCS(), CVXOPT(), GLPK()]

    def test_scalar_lp(self):
        """Test scalar LP problems.
        """
        for solver in self.solvers:
            p = Problem(Minimize(3*self.a), [self.a >= 2])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            result_new = p_new[0].solve(solver.name())
            self.assertAlmostEqual(result, result_new)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            self.assertAlmostEqual(inv_sltn.primal_vars[self.a.id],
                                   self.a.value)

            # TODO: Maximize
            p = Problem(Minimize(-3*self.a + self.b),
                        [self.a <= 2, self.b == self.a, self.b <= 5])
            result = p.solve(solver.name())
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            self.assertAlmostEqual(inv_sltn.primal_vars[self.a.id],
                                   self.a.value)
            self.assertAlmostEqual(inv_sltn.primal_vars[self.b.id],
                                   self.b.value)

            # With a constant in the objective.
            p = Problem(Minimize(3*self.a - self.b + 100),
                        [self.a >= 2,
                         self.b + 5*self.c - 2 == self.a,
                         self.b <= 5 + self.c])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result - 100)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            self.assertAlmostEqual(inv_sltn.primal_vars[self.a.id],
                                   self.a.value)
            self.assertAlmostEqual(inv_sltn.primal_vars[self.b.id],
                                   self.b.value)

            # Unbounded problems.
            # TODO: Maximize
            p = Problem(Minimize(-self.a), [self.a >= 2])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            try:
                result = p.solve(solver.name())
            except SolverError:  # Gurobi fails on this one
                return
            p_new = ConeMatrixStuffing().apply(p)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)

            # Infeasible problems.
            p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
            result = p.solve(solver.name())
            self.assertTrue(FlipObjective().accepts(p))
            p_min = FlipObjective().apply(p)
            self.assertTrue(ConeMatrixStuffing().accepts(p_min[0]))
            p_new = ConeMatrixStuffing().apply(p_min[0])
            result_new = p_new[0].solve(solver.name())
            self.assertAlmostEqual(result, -result_new)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, -result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, -result)
            inv_flipped_sltn = FlipObjective().invert(inv_sltn, p_min[1])
            self.assertAlmostEqual(inv_flipped_sltn.opt_val, result)

    # Test vector LP problems.
    def test_vector_lp(self):
        for solver in self.solvers:
            c = Constant(numpy.array([1, 2]))
            p = Problem(Minimize(c.T*self.x), [self.x >= c])
            result = p.solve(solver.name())
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            p_new = ConeMatrixStuffing().apply(p)
            # result_new = p_new[0].solve(solver.name())
            # self.assertAlmostEqual(result, result_new)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])

            p_new1 = ConeMatrixStuffing().apply(p)
            self.assertTrue(solver.accepts(p_new1[0]))
            sltn = solver.solve(p_new1[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new1[1])

            self.assertAlmostEqual(inv_sltn.opt_val, result)
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[self.x.id],
                                        self.x.value)

            A = Constant(numpy.matrix([[3, 5], [1, 2]]).T).value
            I = Constant([[1, 0], [0, 1]])
            p = Problem(Minimize(c.T*self.x + self.a),
                        [A*self.x >= [-1, 1],
                         4*I*self.z == self.x,
                         self.z >= [2, 2],
                         self.a >= 2])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            result_new = p_new[0].solve(solver.name())
            self.assertAlmostEqual(result, result_new)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result, places=1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=1)

    # Test matrix LP problems.
    def test_matrix_lp(self):
        for solver in self.solvers:
            T = Constant(numpy.ones((2, 2))).value
            p = Problem(Minimize(1 + self.a), [self.A == T + self.a, self.a >= 0])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result - 1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value)

            T = Constant(numpy.ones((2, 3))*2).value
            p = Problem(Minimize(1), [self.A >= T*self.C,
                                      self.A == self.B, self.C == T.T])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result - 1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value)

    def test_socp(self):
        """Test SOCP problems.
        """
        for solver in self.solvers:
            # Basic.
            p = Problem(Minimize(self.b), [pnorm(self.x, p=2) <= self.b])
            pmod = Problem(Minimize(self.b), [SOC(self.b, self.x)])
            self.assertTrue(ConeMatrixStuffing().accepts(pmod))
            p_new = ConeMatrixStuffing().apply(pmod)
            if not solver.accepts(p_new[0]):
                return
            result = p.solve(solver.name())
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value)

            # More complex.
            p = Problem(Minimize(self.b), [pnorm(self.x/2 + self.y[:2], p=2) <= self.b+5,
                                           self.x >= 1, self.y == 5])
            pmod = Problem(Minimize(self.b), [SOC(self.b+5, self.x/2 + self.y[:2]),
                                              self.x >= 1, self.y == 5])
            self.assertTrue(ConeMatrixStuffing().accepts(pmod))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(pmod)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result, places=2)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=2)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=2)

    def exp_cone(self):
        """Test exponential cone problems.
        """
        for solver in self.solvers:
        # Basic.
            p = Problem(Minimize(self.b), [exp(self.a) <= self.b, self.a >= 1])
            pmod = Problem(Minimize(self.b), [ExpCone(self.a, Constant(1), self.b), self.a >= 1])
            self.assertTrue(ConeMatrixStuffing().accepts(pmod))
            p_new = ConeMatrixStuffing().apply(pmod)
            if not solver.accepts(p_new[0]):
                return
            result = p.solve(solver.name())
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result, places=1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
            for var in pmod.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=1)

            # More complex.
            # TODO CVXOPT fails here.
            if solver.name() == 'CVXOPT':
                return
            p = Problem(Minimize(self.b), [exp(self.a/2 + self.c) <= self.b+5,
                                           self.a >= 1, self.c >= 5])
            pmod = Problem(Minimize(self.b), [ExpCone(self.a/2 + self.c, Constant(1), self.b+5),
                                              self.a >= 1, self.c >= 5])
            self.assertTrue(ConeMatrixStuffing().accepts(pmod))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(pmod)
            sltn = solver.solve(p_new[0], False, False, {})
            self.assertAlmostEqual(sltn.opt_val, result, places=0)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=0)
            for var in pmod.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=0)

    # Test positive definite constraints.
    def test_psd_constraints(self):
        """ Test positive semi-definite constraints
        """
        C = Variable((3, 3))
        obj = Maximize(C[0, 2])
        constraints = [diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == C.T,
                       C >> 0]
        prob = Problem(obj, constraints)
        self.assertTrue(FlipObjective().accepts(prob))
        p_min = FlipObjective().apply(prob)
        self.assertTrue(ConeMatrixStuffing().accepts(p_min[0]))

        C = Variable((2, 2))
        obj = Maximize(C[0, 1])
        constraints = [C == 1, C >> [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        self.assertTrue(FlipObjective().accepts(prob))
        p_min = FlipObjective().apply(prob)
        self.assertTrue(ConeMatrixStuffing().accepts(p_min[0]))

        C = Variable((2, 2), symmetric=True)
        obj = Minimize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob, _ = CvxAttr2Constr().apply(Problem(obj, constraints))
        self.assertTrue(ConeMatrixStuffing().accepts(prob))
