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
from cvxpy.expressions.variables import Variable, Semidef, Bool, Symmetric
from cvxpy.constraints import SOC, ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy import Problem, Minimize, Maximize
from cvxpy.tests.base_test import BaseTest
from cvxpy.reductions.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.solver_interface.conic_solvers.ecos_conif import ECOS
from cvxpy.solver_interface.conic_solvers.scs_conif import SCS
import numpy

class TestLinearCone(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

        self.solver = ECOS()
        self.solver_name = 'ECOS'

    def test_scalar_lp(self):
        """Test scalar LP problems.
        """
        p = Problem(Minimize(3*self.a), [self.a >= 2])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, result_new)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)
        self.assertAlmostEqual(inv_sltn.primal_vars[self.a.id],
                               self.a.value)

        p = Problem(Maximize(3*self.a - self.b),
                    [self.a <= 2, self.b == self.a, self.b <= 5])
        result = p.solve(self.solver_name)
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)
        self.assertAlmostEqual(p_new[0].solve(self.solver_name), -result)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, -result)
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
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, result_new)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)
        self.assertAlmostEqual(inv_sltn.primal_vars[self.a.id],
                               self.a.value)
        self.assertAlmostEqual(inv_sltn.primal_vars[self.b.id],
                               self.b.value)

        # Unbounded problems.
        p = Problem(Maximize(self.a), [self.a >= 2])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, -result_new)
        self.assertTrue(self.solver.accepts(p_new[0]))
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, -result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)

        # Infeasible problems.
        p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, -result_new)
        self.assertTrue(self.solver.accepts(p_new[0]))
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, -result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)

    # Test vector LP problems.
    def test_vector_lp(self):
        c = Constant(numpy.matrix([1, 2]).T).value
        p = Problem(Minimize(c.T*self.x), [self.x >= c])
        result = p.solve(self.solver_name)
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        p_new = ConeMatrixStuffing().apply(p)
        # result_new = p_new[0].solve(self.solver_name)
        # self.assertAlmostEqual(result, result_new)
        self.assertTrue(self.solver.accepts(p_new[0]))
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])

        p_new1 = ConeMatrixStuffing().apply(p)
        self.assertTrue(self.solver.accepts(p_new1[0]))
        sltn = self.solver.solve(p_new1[0], False, False, {})
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
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, result_new)
        self.assertTrue(self.solver.accepts(p_new[0]))
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result, places=1)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
        for var in p.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value, places=1)

    # Test matrix LP problems.
    def test_matrix_lp(self):
        T = Constant(numpy.ones((2, 2))).value
        p = Problem(Minimize(1), [self.A == T])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, result_new)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)
        for var in p.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value)

        T = Constant(numpy.ones((2, 3))*2).value
        c = Constant(numpy.matrix([3, 4]).T).value
        p = Problem(Minimize(1), [self.A >= T*self.C,
                                  self.A == self.B, self.C == T.T])
        self.assertTrue(ConeMatrixStuffing().accepts(p))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(p)
        result_new = p_new[0].solve(self.solver_name)
        self.assertAlmostEqual(result, result_new)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)
        for var in p.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value)

    def test_socp(self):
        """Test SOCP problems.
        """
        # Basic.
        p = Problem(Minimize(self.b), [norm2(self.x) <= self.b])
        pmod = Problem(Minimize(self.b), [SOC(self.b, self.x)])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(pmod)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result)
        for var in p.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value)

        # More complex.
        p = Problem(Minimize(self.b), [norm2(self.x/2 + self.y[:2]) <= self.b+5,
                                       self.x >= 1, self.y == 5])
        pmod = Problem(Minimize(self.b), [SOC(self.b+5, self.x/2 + self.y[:2]),
                                          self.x >= 1, self.y == 5])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(pmod)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result, places=2)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result, places=2)
        for var in p.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value, places=2)

    def test_exp_cone(self):
        """Test exponential cone problems.
        """
        # Basic.
        p = Problem(Minimize(self.b), [exp(self.a) <= self.b, self.a >= 1])
        pmod = Problem(Minimize(self.b), [ExpCone(self.a, Constant(1), self.b), self.a >= 1])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(pmod)
        sltn = self.solver.solve(p_new[0], False, False, {})
        self.assertAlmostEqual(sltn.opt_val, result, places=1)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
        for var in pmod.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                        var.value, places=1)

        # More complex.
        p = Problem(Minimize(self.b), [exp(self.a/2 + self.c) <= self.b+5,
                                       self.a >= 1, self.c == 5])
        pmod = Problem(Minimize(self.b), [ExpCone(self.a/2 + self.c, Constant(1), self.b+5),
                                          self.a >= 1, self.c == 5])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        result = p.solve(self.solver_name)
        p_new = ConeMatrixStuffing().apply(pmod)
        sltn = self.solver.solve(p_new[0], False, False, {})
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
        C = Variable(3, 3)
        obj = Maximize(C[0, 2])
        constraints = [diag(C) == 1,
                       C[0, 1] == 0.6,
                       C[1, 2] == -0.3,
                       C == C.T,
                       C >> 0]
        prob = Problem(obj, constraints)
        self.assertTrue(ConeMatrixStuffing().accepts(prob))
        prob_new = ConeMatrixStuffing().apply(prob)

        C = Variable(2, 2)
        obj = Maximize(C[0, 1])
        constraints = [C == 1, C >> [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        self.assertTrue(ConeMatrixStuffing().accepts(prob))
        prob_new = ConeMatrixStuffing().apply(prob)

        C = Symmetric(2, 2)
        obj = Minimize(C[0, 0])
        constraints = [C << [[2, 0], [0, 2]]]
        prob = Problem(obj, constraints)
        self.assertTrue(ConeMatrixStuffing().accepts(prob))
        prob_new = ConeMatrixStuffing().apply(prob)
