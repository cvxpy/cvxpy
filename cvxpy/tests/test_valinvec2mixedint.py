#!/usr/bin/env python
"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH

solver_installed = pytest.mark.skipif(
    cp.GLPK_MI not in INSTALLED_SOLVERS, reason="Required solver not installed"
)


@solver_installed
@pytest.mark.parametrize("ineq_form", [True, False])
class TestFiniteSet:
    @staticmethod
    def make_test_1(ineq_form: bool):
        """vec contains a contiguous range of integers"""
        x = cp.Variable(shape=(4,))
        expect_x = np.array([0., 7., 3., 0.])
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        constr1 = FiniteSet(x[0], vec, ineq_form=ineq_form)
        constr2 = FiniteSet(x[1], vec, ineq_form=ineq_form)
        constr3 = FiniteSet(x[2], vec, ineq_form=ineq_form)
        constr4 = FiniteSet(x[3], vec, ineq_form=ineq_form)
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

    def test_1(self, ineq_form: bool):
        sth = TestFiniteSet.make_test_1(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_2(ineq_form: bool):
        x = cp.Variable()
        expect_x = np.array([-1.125])
        objective = cp.Minimize(x)
        vec = [-1.125, 1, 2]
        constr1 = x >= -1.25
        constr2 = x <= 10
        constr3 = FiniteSet(x, vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_2(ineq_form: bool):
        sth = TestFiniteSet.make_test_2(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_3(ineq_form: bool):
        """Case when vec.size==1"""
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x - 3))
        vec = [1]
        cons1 = FiniteSet(x, vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_3(ineq_form: bool):
        sth = TestFiniteSet.make_test_3(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_4(ineq_form: bool):
        """Case when vec houses duplicates"""
        x = cp.Variable()
        objective = cp.Minimize(cp.abs(x - 3))
        vec = [1, 1, 1, 2, 2, 3, 3]
        cons1 = FiniteSet(x, vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_4(ineq_form: bool):
        sth = TestFiniteSet.make_test_4(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_5(ineq_form: bool):
        """Case when input expression to FiniteSet constraint is affine"""
        x = cp.Variable(shape=(4,))
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        expr0 = 2 * x[0] + 1
        expr2 = 3 * x[2] + 5
        constr1 = FiniteSet(expr0, vec, ineq_form=ineq_form)
        constr2 = FiniteSet(x[1], vec, ineq_form=ineq_form)
        constr3 = FiniteSet(expr2, vec, ineq_form=ineq_form)
        constr4 = FiniteSet(x[3], vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_5(ineq_form: bool):
        sth = TestFiniteSet.make_test_5(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_6(ineq_form: bool):
        """vec contains only real quantities + passed expression is affine"""
        x = cp.Variable()
        expect_x = np.array([-1.0625])
        objective = cp.Minimize(x)
        vec = [-1.125, 1.5, 2.24]
        constr1 = x >= -1.25
        constr2 = x <= 10
        expr = 2 * x + 1
        constr3 = FiniteSet(expr, vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_6(ineq_form: bool):
        sth = TestFiniteSet.make_test_6(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_7(ineq_form: bool):
        """For testing vectorization of FiniteSet class"""
        x = cp.Variable(shape=(4,))
        expect_x = np.array([0., 7., 3., 0.])
        vec = np.arange(10)
        objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
        constr1 = FiniteSet(x, vec, ineq_form=ineq_form)
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

    @staticmethod
    def test_7(ineq_form: bool):
        sth = TestFiniteSet.make_test_7(ineq_form)
        sth.solve(solver='GLPK_MI')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def test_8(ineq_form: bool):
        # Test parametrized FiniteSet
        x = cp.Variable()
        objective = cp.Maximize(x)
        set_vals = cp.Parameter((5,), value=np.arange(5))
        constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form)]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, 4)

        set_vals.value = np.arange(5) + 1
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, 5)

    @staticmethod
    def test_9(ineq_form: bool):
        # Test passing a Python set
        x = cp.Variable()
        objective = cp.Maximize(x)
        set_vals = set(range(5))
        constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form)]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, 4)

    @staticmethod
    def test_10(ineq_form: bool):
        # Test set with two elements
        x = cp.Variable()
        objective = cp.Maximize(x)
        set_vals = {1, 2}
        constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form)]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, 2)

    @staticmethod
    def test_11(ineq_form: bool):
        # Test 2D Variable
        shape = (2, 2)
        x = cp.Variable(shape)
        objective = cp.Maximize(cp.sum(x))
        set_vals = {1, 2, 3}
        constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form)]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, np.ones(shape)*max(set_vals))

    @staticmethod
    def test_non_affine_exception(ineq_form: bool):
        # Exception test: non-affine expression
        x = cp.Variable()
        x_abs = cp.abs(x)
        set_vals = {1, 2, 3}
        with pytest.raises(ValueError, match="must be affine"):
            FiniteSet(x_abs, set_vals, ineq_form=ineq_form)

    @staticmethod
    def test_independent_entries(ineq_form: bool):
        shape = (2, 2)
        x = cp.Variable(shape)
        objective = cp.Maximize(cp.sum(x))
        set_vals = {0, 1, 2}
        constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form),
                       x <= np.arange(4).reshape(shape)]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GLPK_MI)
        assert np.allclose(x.value, np.array([[0, 1], [2, 2]]))


@solver_installed
def test_default_argument():
    x = cp.Variable()
    objective = cp.Maximize(x)
    set_vals = set(range(5))
    constraints = [FiniteSet(x, set_vals)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    assert np.allclose(x.value, 4)
