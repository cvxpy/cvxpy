#!/usr/bin/env python
"""
Copyright, the CVXPY authors

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

import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class TestFiniteSet(BaseTest):
    @staticmethod
    def make_test_1():
        """vec contains a contiguous range of integers"""
        x = cp.Variable(shape=(4,))
        expect_x = np.array([0., 7., 3., 0.])
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        constr1 = FiniteSet(x[0], vec)
        constr2 = FiniteSet(x[1], vec)
        constr3 = FiniteSet(x[2], vec)
        constr4 = FiniteSet(x[3], vec)
        constr5 = x[0] + 2 * x[2] <= 700
        constr6 = 2 * x[1] - 8 * x[2] <= 0
        constr7 = x[1] - 2 * x[2] + x[3] >= 1
        constr8 = x[0] + x[1] + x[2] + x[3] == 10
        obj_pair = (objective, 13.0)
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None),
            (constr4, None),
            (constr5, None),
            (constr6, None),
            (constr7, None),
            (constr8, None)
        ]
        var_pairs = [
            (x, expect_x)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_1(self):
        sth = TestFiniteSet.make_test_1()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_2():
        x = cp.Variable()
        expect_x = np.array([-1.125])
        objective = cp.Minimize(x)
        vec = [-1.125, 1, 2]
        constr1 = x >= -1.25
        constr2 = x <= 10
        constr3 = FiniteSet(x, vec)
        obj_pairs = (objective, -1.125)
        var_pairs = [
            (x, expect_x)
        ]
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None)
        ]
        sth = STH.SolverTestHelper(obj_pairs, var_pairs, con_pairs)
        return sth

    def test_2(self):
        sth = TestFiniteSet.make_test_2()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_3():
        """Case when vec.size==1"""
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x - 3))
        vec = [1]
        cons1 = FiniteSet(x, vec)
        expected_x = np.array([1.])
        obj_pair = (objective, 2.0)
        var_pairs = [
            (x, expected_x)
        ]
        con_pairs = [
            (cons1, None)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_3(self):
        sth = TestFiniteSet.make_test_3()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_4():
        """Case when vec houses duplicates"""
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x - 3))
        vec = [1, 1, 1, 2, 2, 3, 3]
        cons1 = FiniteSet(x, vec)
        expected_x = np.array([3.])
        obj_pair = (objective, 0.0)
        var_pairs = [
            (x, expected_x)
        ]
        con_pairs = [
            (cons1, None)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_4(self):
        sth = TestFiniteSet.make_test_4()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_5():
        """Case when input expression to FiniteSet constraint is affine"""
        x = cp.Variable(shape=(4,))
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        expr0 = 2 * x[0] + 1
        expr2 = 3 * x[2] + 5
        constr1 = FiniteSet(expr0, vec)
        constr2 = FiniteSet(x[1], vec)
        constr3 = FiniteSet(expr2, vec)
        constr4 = FiniteSet(x[3], vec)
        constr5 = x[0] + 2 * x[2] <= 700
        constr6 = 2 * x[1] - 8 * x[2] <= 0
        constr7 = x[1] - 2 * x[2] + x[3] >= 1
        constr8 = x[0] + x[1] + x[2] + x[3] == 10
        expected_x = np.array([4., 4., 1., 1.])
        obj_pair = (objective, 8.0)
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None),
            (constr4, None),
            (constr5, None),
            (constr6, None),
            (constr7, None),
            (constr8, None)
        ]
        var_pairs = [
            (x, expected_x)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_5(self):
        sth = TestFiniteSet.make_test_5()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_6():
        """vec contains only real quantities + passed expression is affine"""
        x = cp.Variable()
        expect_x = np.array([-1.0625])
        objective = cp.Minimize(x)
        vec = [-1.125, 1.5, 2.24]
        constr1 = x >= -1.25
        constr2 = x <= 10
        expr = 2 * x + 1
        constr3 = FiniteSet(expr, vec)
        obj_pairs = (objective, -1.0625)
        var_pairs = [
            (x, expect_x)
        ]
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None)
        ]
        sth = STH.SolverTestHelper(obj_pairs, var_pairs, con_pairs)
        return sth

    def test_6(self):
        sth = TestFiniteSet.make_test_6()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_7():
        """For testing vectorization of FiniteSet class"""
        x = cp.Variable(shape=(4,))
        expect_x = np.array([0., 7., 3., 0.])
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        constr1 = FiniteSet(x, vec, ineq_form=False)
        constr2 = x[0] + 2 * x[2] <= 700
        constr3 = 2 * x[1] - 8 * x[2] <= 0
        constr4 = x[1] - 2 * x[2] + x[3] >= 1
        constr5 = x[0] + x[1] + x[2] + x[3] == 10
        obj_pair = (objective, 13.0)
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None),
            (constr4, None),
            (constr5, None)
        ]
        var_pairs = [
            (x, expect_x)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_7(self):
        sth = TestFiniteSet.make_test_7()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_8():
        """Testing the alternative constraining pathway"""
        x = cp.Variable()
        expect_x = np.array([-1.125])
        objective = cp.Minimize(x)
        vec = [-1.125, 1, 2]
        constr1 = x >= -1.25
        constr2 = x <= 10
        constr3 = FiniteSet(x, vec, ineq_form=True)
        obj_pairs = (objective, -1.125)
        var_pairs = [
            (x, expect_x)
        ]
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None)
        ]
        sth = STH.SolverTestHelper(obj_pairs, var_pairs, con_pairs)
        return sth

    def test_8(self):
        sth = TestFiniteSet.make_test_8()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass

    @staticmethod
    def make_test_9():
        """For testing vectorization of FiniteSet class + new constraining method"""
        x = cp.Variable(shape=(4,))
        expect_x = np.array([0., 7., 3., 0.])
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        constr1 = FiniteSet(x, vec, ineq_form=True)
        constr2 = x[0] + 2 * x[2] <= 700
        constr3 = 2 * x[1] - 8 * x[2] <= 0
        constr4 = x[1] - 2 * x[2] + x[3] >= 1
        constr5 = x[0] + x[1] + x[2] + x[3] == 10
        obj_pair = (objective, 13.0)
        con_pairs = [
            (constr1, None),
            (constr2, None),
            (constr3, None),
            (constr4, None),
            (constr5, None)
        ]
        var_pairs = [
            (x, expect_x)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_9(self):
        sth = TestFiniteSet.make_test_9()
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        pass
