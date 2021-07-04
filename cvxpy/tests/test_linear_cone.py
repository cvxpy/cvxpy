"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from cvxpy import Maximize, Minimize, Problem
from cvxpy.atoms import diag, exp, hstack, pnorm
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import SolverTestHelper


def solve_wrapper(solver, param_cone_prog):
    data, inv_data = solver.apply(param_cone_prog)
    return solver.invert(solver.solve_via_data(
        data, warm_start=False, verbose=False, solver_opts={}), inv_data)


class TestLinearCone(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self) -> None:
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

        self.solvers = [ECOS()]

    def test_scalar_lp(self) -> None:
        """Test scalar LP problems.
        """
        for solver in self.solvers:
            p = Problem(Minimize(3*self.a), [self.a >= 2])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(result, sltn.opt_val)
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
            sltn = solve_wrapper(solver, p_new[0])
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
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result)
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
            sltn = solve_wrapper(solver, p_new[0])
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
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, -result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, -result)
            inv_flipped_sltn = FlipObjective().invert(inv_sltn, p_min[1])
            self.assertAlmostEqual(inv_flipped_sltn.opt_val, result)

    # Test vector LP problems.
    def test_vector_lp(self) -> None:
        for solver in self.solvers:
            c = Constant(np.array([1, 2]))
            p = Problem(Minimize(c.T @ self.x), [self.x >= c])
            result = p.solve(solver.name())
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            p_new = ConeMatrixStuffing().apply(p)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])

            p_new1 = ConeMatrixStuffing().apply(p)
            self.assertTrue(solver.accepts(p_new1[0]))
            sltn = solve_wrapper(solver, p_new1[0])
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new1[1])

            self.assertAlmostEqual(inv_sltn.opt_val, result)
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[self.x.id],
                                        self.x.value)

            A = Constant(np.array([[3, 5], [1, 2]]).T).value
            Imat = Constant([[1, 0], [0, 1]])
            p = Problem(Minimize(c.T @ self.x + self.a),
                        [A @ self.x >= [-1, 1],
                         4*Imat @ self.z == self.x,
                         self.z >= [2, 2],
                         self.a >= 2])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            self.assertTrue(solver.accepts(p_new[0]))
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result, places=1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=1)

    # Test matrix LP problems.
    def test_matrix_lp(self) -> None:
        for solver in self.solvers:
            T = Constant(np.ones((2, 2))).value
            p = Problem(Minimize(self.a), [self.A == T + self.a, self.a >= 0])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value)

            T = Constant(np.ones((2, 3))*2).value
            p = Problem(Minimize(1), [self.A >= T @ self.C,
                                      self.A == self.B, self.C == T.T])
            self.assertTrue(ConeMatrixStuffing().accepts(p))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(p)
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value)

    def test_socp(self) -> None:
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
            sltn = solve_wrapper(solver, p_new[0])
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
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result, places=2)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=2)
            for var in p.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=2)

    def exp_cone(self) -> None:
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
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result, places=1)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
            for var in pmod.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=1)

            # More complex.
            p = Problem(Minimize(self.b), [exp(self.a/2 + self.c) <= self.b+5,
                                           self.a >= 1, self.c >= 5])
            pmod = Problem(Minimize(self.b), [ExpCone(self.a/2 + self.c, Constant(1), self.b+5),
                                              self.a >= 1, self.c >= 5])
            self.assertTrue(ConeMatrixStuffing().accepts(pmod))
            result = p.solve(solver.name())
            p_new = ConeMatrixStuffing().apply(pmod)
            sltn = solve_wrapper(solver, p_new[0])
            self.assertAlmostEqual(sltn.opt_val, result, places=0)
            inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
            self.assertAlmostEqual(inv_sltn.opt_val, result, places=0)
            for var in pmod.variables():
                self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id],
                                            var.value, places=0)

    # Test positive definite constraints.
    def test_psd_constraints(self) -> None:
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

    def test_nonneg_constraints_backend(self) -> None:
        x = Variable(shape=(2,), name='x')
        objective = Maximize(-4 * x[0] - 5 * x[1])
        constr_expr = hstack([3 - (2 * x[0] + x[1]),
                              3 - (x[0] + 2 * x[1]),
                              x[0],
                              x[1]])
        constraints = [NonNeg(constr_expr)]
        prob = Problem(objective, constraints)
        self.assertFalse(ConeMatrixStuffing().accepts(prob))
        self.assertTrue(FlipObjective().accepts(prob))
        p_min = FlipObjective().apply(prob)
        self.assertTrue(ConeMatrixStuffing().accepts(p_min[0]))

    def test_nonneg_constraints_end_user(self) -> None:
        x = Variable(shape=(2,), name='x')
        objective = Minimize(-4 * x[0] - 5 * x[1])
        constr_expr = hstack([3 - (2 * x[0] + x[1]),
                              3 - (x[0] + 2 * x[1]),
                              x[0],
                              x[1]])
        constraints = [NonNeg(constr_expr)]
        expect_dual_var = np.array([1, 2, 0, 0])
        con_pairs = [(constraints[0], expect_dual_var)]
        var_pairs = [(x, np.array([1, 1]))]
        obj_pair = (objective, -9)
        # Check that the problem compiles correctly, and that
        # dual variables are recovered correctly.
        sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
        sth.solve(solver='ECOS')
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)
        # Check that violations are computed properly
        expr_val = constr_expr.value  # want >= 0
        expr_val[expr_val >= 0] = 0
        manual_viol = np.linalg.norm(expr_val, ord=2)
        reported_viol = constraints[0].violation()
        self.assertAlmostEqual(manual_viol, reported_viol, places=4)
        # Check that residuals are computed properly
        x.value = np.array([-1, -2])
        expr_val = constraints[0].residual
        self.assertItemsAlmostEqual(
            expr_val,  # first two constraints are feasible.
            np.array([0, 0, 1, 2])
        )
        # Run a second check for violations
        reported_viol = constraints[0].violation()
        expected_viol = np.sqrt(5.0)
        self.assertAlmostEqual(reported_viol, expected_viol)
