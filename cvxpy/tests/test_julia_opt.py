"""
Copyright 2017 Steven Diamond

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

from cvxpy import *
import cvxpy.atoms.elementwise.log as cvxlog
from cvxpy.tests.base_test import BaseTest
import unittest
import math
import numpy as np
import sys
if sys.version_info >= (3, 0):
    from functools import reduce

class TestJuliaOpt(BaseTest):
    """ Unit tests for the Julia opt interface. """
    OPTIONS = [("ECOS", "ECOSSolver(verbose=1)"), ("SCS", "SCSSolver(verbose=1)")]
    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Overriden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestJuliaOpt, self).assertItemsAlmostEqual(a,b,places=places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestJuliaOpt, self).assertAlmostEqual(a, b, places=places)

    # Test scalar LP problems.
    def test_scalar_lp(self):
        if JULIA_OPT in installed_solvers():
            p = Problem(Minimize(3*self.a), [self.a >= 2])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 6)
                self.assertAlmostEqual(self.a.value, 2)

            p = Problem(Maximize(3*self.a - self.b),
                [self.a <= 2, self.b == self.a, self.b <= 5])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 4.0)
                self.assertAlmostEqual(self.a.value, 2)
                self.assertAlmostEqual(self.b.value, 2)

            # With a constant in the objective.
            p = Problem(Minimize(3*self.a - self.b + 100),
                [self.a >= 2,
                self.b + 5*self.c - 2 == self.a,
                self.b <= 5 + self.c])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 101 + 1.0/6)
                self.assertAlmostEqual(self.a.value, 2)
                self.assertAlmostEqual(self.b.value, 5-1.0/6)
                self.assertAlmostEqual(self.c.value, -1.0/6)

            # Test status and value.
            exp = Maximize(self.a)
            p = Problem(exp, [self.a <= 2])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertEqual(result, p.value)
                self.assertEqual(p.status, OPTIMAL)
                assert self.a.value is not None
                assert p.constraints[0].dual_value is not None

            # Unbounded problems.
            p = Problem(Maximize(self.a), [self.a >= 2])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertEqual(p.status, UNBOUNDED)
                assert np.isinf(p.value)
                assert p.value > 0
                assert self.a.value is None
                assert p.constraints[0].dual_value is None

            p = Problem(Minimize(-self.a), [self.a >= 2])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertEqual(result, p.value)
                self.assertEqual(p.status, UNBOUNDED)
                assert np.isinf(p.value)
                assert p.value < 0

            # Infeasible problem
            p = Problem(Maximize(self.a), [self.a >= 2, self.a <= 1])
            self.a.save_value(2)
            p.constraints[0].save_value(2)

            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertEqual(result, p.value)
                self.assertEqual(p.status, INFEASIBLE)
                assert np.isinf(p.value)
                assert p.value < 0
                assert self.a.value is None
                assert p.constraints[0].dual_value is None

            p = Problem(Minimize(-self.a), [self.a >= 2, self.a <= 1])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertEqual(result, p.value)
                self.assertEqual(p.status, INFEASIBLE)
                assert np.isinf(p.value)
                assert p.value > 0

        # Test vector LP problems.
        def test_vector_lp(self):
            c = Constant([1,2])
            p = Problem(Minimize(c.T*self.x), [self.x >= c])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 5)
                self.assertItemsAlmostEqual(self.x.value, [1,2])

            A = Constant([[3,5],[1,2]])
            I = Constant([[1,0],[0,1]])
            p = Problem(Minimize(c.T*self.x + self.a),
                [A*self.x >= [-1,1],
                4*I*self.z == self.x,
                self.z >= [2,2],
                self.a >= 2])
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 26, places=3)
                obj = c.T*self.x + self.a
                self.assertAlmostEqual(obj.value, result)
                self.assertItemsAlmostEqual(self.x.value, [8,8], places=3)
                self.assertItemsAlmostEqual(self.z.value, [2,2], places=3)

    def test_log_problem(self):
        if JULIA_OPT in installed_solvers():
            # Log in objective.
            obj = Maximize(sum_entries(log(self.x)))
            constr = [self.x <= [1, math.e]]
            p = Problem(obj, constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 1)
                self.assertItemsAlmostEqual(self.x.value, [1, math.e])

            # Log in constraint.
            obj = Minimize(sum_entries(self.x))
            constr = [log(self.x) >= 0, self.x <= [1,1]]
            p = Problem(obj, constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 2)
                self.assertItemsAlmostEqual(self.x.value, [1,1])

            # Index into log.
            obj = Maximize(log(self.x)[1])
            constr = [self.x <= [1, math.e]]
            p = Problem(obj,constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 1)

    def test_sqrt_problem(self):
        if JULIA_OPT in installed_solvers():
            # sqrt in objective.
            obj = Maximize(sum_entries(sqrt(self.x)))
            constr = [self.x <= [1, 4]]
            p = Problem(obj, constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 3)
                self.assertItemsAlmostEqual(self.x.value, [1, 4])

            # sqrt in constraint.
            obj = Minimize(sum_entries(self.x))
            constr = [sqrt(self.x) >= [2,3]]
            p = Problem(obj, constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 13)
                self.assertItemsAlmostEqual(self.x.value, [4,9])

            # Index into sqrt.
            obj = Maximize(sqrt(self.x)[1])
            constr = [self.x <= [1, 4]]
            p = Problem(obj,constr)
            for pkg, solver_str in self.OPTIONS:
                result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
                self.assertAlmostEqual(result, 2)

    def test_pajarito(self):
        """Test mixed integer problem with Pajarito.
        """
        if JULIA_OPT in installed_solvers():
            pkg, solver_str = ("Pajarito, GLPKMathProgInterface, ECOS", "PajaritoSolver(verbose=1,mip_solver=GLPKSolverMIP(),cont_solver=ECOSSolver(verbose=0))")
            # sqrt in constraint.
            x_int = Int(2)
            obj = Minimize(sum_entries(x_int))
            constr = [sqrt(x_int) >= [2,3]]
            p = Problem(obj, constr)
            result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
            self.assertAlmostEqual(result, 13)
            self.assertItemsAlmostEqual(x_int.value, [4,9])

            x_bool = mul_elemwise([4,9], Bool(2))
            obj = Minimize(sum_entries(x_bool))
            constr = [sqrt(x_bool) >= [2,3]]
            p = Problem(obj, constr)
            result = p.solve(solver=JULIA_OPT, package=pkg, solver_str=solver_str)
            self.assertAlmostEqual(result, 13)
            self.assertItemsAlmostEqual(x_bool.value, [4,9])
