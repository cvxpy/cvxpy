"""
Copyright 2022, the CVXPY developers.

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

from itertools import product

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import scopes


class TestKron(BaseTest):
    """
    The Kronecker product of matrices M, N is :

        kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
                     [M[1,0] * N   , ..., M[1, end] * N  ]
                     ...
                     [M[end, 0] * N, ..., M[end, end] * N]
    """

    @staticmethod
    def make_kron_prob(z_dims: tuple[int],
                       c_dims: tuple[int],
                       param: bool,
                       var_left: bool,
                       seed: int):
        """
        Construct random nonnegative matrices (C, L) of shapes
        (c_dims, z_dims) respectively. Define an optimization
        problem with a matrix variable of shape z_dims:

            min sum(Z)
            s.t.  kron(Z, C) >= kron(L, C)   ---   if var_left is True
                  kron(C, Z) >= kron(C, L)   ---   if var_left is False
                  Z >= 0

        Regardless of whether var_left is True or False, the optimal
        solution to that problem is Z = L.

        If param is True, then C is defined as a CVXPY Parameter.
        If param is False, then C is a CVXPY Constant.

        A small remark: the constraint that Z >= 0 is redundant.
        It's there because it's easier to set break points that distinguish
        objective canonicalization and constraint canonicalization
        when there's more than one constraint.
        """
        np.random.seed(seed)
        C_value = np.random.rand(*c_dims).round(decimals=2)
        if param:
            C = cp.Parameter(shape=c_dims)
            C.value = C_value
        else:
            C = cp.Constant(C_value)
        Z = cp.Variable(shape=z_dims)
        L = np.random.rand(*Z.shape).round(decimals=2)
        if var_left:
            constraints = [cp.kron(Z, C) >= cp.kron(L, C), Z >= 0]
        else:
            constraints = [cp.kron(C, Z) >= cp.kron(C, L), Z >= 0]
        obj_expr = cp.sum(Z)
        prob = cp.Problem(cp.Minimize(obj_expr), constraints)
        return Z, C, L, prob


class TestKronRightVar(TestKron):

    C_DIMS = [(1, 1), (2, 1), (1, 2), (2, 2)]

    def test_gen_kronr_param(self):
        z_dims = (2, 2)
        for c_dims in TestKronRightVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=True,
                                                var_left=False, seed=0)
            prob.solve(solver=cp.CLARABEL)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)

    def test_gen_kronr_const(self):
        z_dims = (2, 2)
        for c_dims in TestKronRightVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=False,
                                                var_left=False, seed=0)
            prob.solve(solver=cp.CLARABEL)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)


class TestKronLeftVar(TestKron):

    C_DIMS = [(1, 1), (2, 1), (1, 2), (2, 2)]

    def symvar_kronl(self, param):
        # Use a symmetric matrix variable
        X = cp.Variable(shape=(2, 2), symmetric=True)
        b_val = 1.5 * np.ones((1, 1))
        if param:
            b = cp.Parameter(shape=(1, 1))
            b.value = b_val
        else:
            b = cp.Constant(b_val)
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        kronX = cp.kron(X, b)  # should be equal to X

        objective = cp.Minimize(cp.sum(X.flatten(order='F')))
        constraints = [U >= kronX, kronX >= L]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.assertItemsAlmostEqual(X.value, np.array([[0.5, 2], [2, 3]]) / 1.5)
        objective = cp.Maximize(cp.sum(X.flatten(order='F')))
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(X.value, np.array([[10, 11], [11, 13]]) / 1.5)
        pass

    def test_symvar_kronl_param(self):
        self.symvar_kronl(param=True)

    def test_symvar_kronl_const(self):
        self.symvar_kronl(param=False)

    def scalar_kronl(self, param):
        y = cp.Variable(shape=(1, 1))
        A_val = np.array([[1., 2.], [3., 4.]])
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        if param:
            A = cp.Parameter(shape=(2, 2))
            A.value = A_val
        else:
            A = cp.Constant(A_val)
        krony = cp.kron(y, A)  # should be equal to y * A
        constraints = [U >= krony, krony >= L]

        objective = cp.Minimize(y)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(y.value, np.array([[np.max(L / A_val)]]))

        objective = cp.Maximize(y)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(y.value, np.array([[np.min(U / A_val)]]))
        pass

    def test_scalar_kronl_param(self):
        self.scalar_kronl(param=True)

    def test_scalar_kronl_const(self):
        self.scalar_kronl(param=False)

    def test_gen_kronl_param(self):
        z_dims = (2, 2)
        for c_dims in TestKronLeftVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=True,
                                                var_left=True, seed=0)
            prob.solve(solver=cp.CLARABEL)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)

    def test_gen_kronr_const(self):
        z_dims = (2, 2)
        for c_dims in TestKronLeftVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=False,
                                                var_left=True, seed=0)
            prob.solve(solver=cp.CLARABEL)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)


class TestKronDPP(TestKron):
    BACKENDS = [cp.CPP_CANON_BACKEND, cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]

    def test_parameter_coefficients(self):
        for backend, var_left, shape in product(
            self.BACKENDS, [False, True], [(1, 1), (2, 1), (1, 2), (2, 3)]
        ):
            with self.subTest(backend=backend, var_left=var_left, shape=shape):
                Z, C, L, problem = self.make_kron_prob(
                    (2, 2), shape, param=True, var_left=var_left, seed=0
                )
                self.assertTrue(problem.is_dpp())
                initial = C.value.copy()
                cached = None
                for scale in [1.0, 2.0, 0.5]:
                    C.value = scale * initial
                    problem.solve(
                        solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True
                    )
                    self.assertEqual(problem.status, cp.OPTIMAL)
                    self.assertItemsAlmostEqual(Z.value, L, places=5)
                    self.assertIsNotNone(problem._cache.param_prog)
                    if cached is None:
                        cached = problem._cache.param_prog
                    self.assertIs(problem._cache.param_prog, cached)

    def test_zero_parameter_id(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = np.array([[0.5, -2.0], [3.0, 1.0]])
        for backend, param_left in product(self.BACKENDS, [False, True]):
            with self.subTest(backend=backend, param_left=param_left):
                parameter = cp.Parameter(values.shape, id=0, value=values)
                variable = cp.Variable(target.shape)
                if param_left:
                    expression = cp.kron(parameter, variable)
                    expected = np.kron(values, target)
                else:
                    expression = cp.kron(variable, parameter)
                    expected = np.kron(target, values)
                problem = cp.Problem(cp.Minimize(cp.sum_squares(expression - expected)))
                problem.solve(solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True)
                self.assertItemsAlmostEqual(variable.value, target, places=5)
                self.assertAlmostEqual(problem.value, 0.0, places=7)

    def test_parameters_in_variable_operand(self):
        constant = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        values = np.array([[0.5, -2.0], [3.0, 1.0]])
        for backend, param_left in product(self.BACKENDS, [False, True]):
            with self.subTest(backend=backend, param_left=param_left):
                parameter = cp.Parameter(values.shape, value=values)
                variable = cp.Variable(values.shape)
                operand = variable + parameter
                expression = (
                    cp.kron(operand, constant) if param_left else cp.kron(constant, operand)
                )
                problem = cp.Problem(cp.Minimize(cp.sum_squares(expression)))
                self.assertTrue(problem.is_dpp())
                for scale in [1.0, -2.0]:
                    parameter.value = scale * values
                    problem.solve(
                        solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True
                    )
                    self.assertItemsAlmostEqual(variable.value, -parameter.value, places=5)

    def test_zero_parameter_updates(self):
        target = np.array([[0.5, 2.0], [3.0, 1.0]])
        for backend, param_left in product(self.BACKENDS, [False, True]):
            with self.subTest(backend=backend, param_left=param_left):
                parameter = cp.Parameter((2, 3), nonneg=True)
                variable = cp.Variable(target.shape, nonneg=True)
                if param_left:
                    lhs, rhs = cp.kron(parameter, variable), cp.kron(parameter, target)
                else:
                    lhs, rhs = cp.kron(variable, parameter), cp.kron(target, parameter)
                problem = cp.Problem(cp.Minimize(cp.sum(variable)), [lhs >= rhs])
                cached = None
                for scale in [1.0, 0.0, 2.0]:
                    parameter.value = np.full((2, 3), scale)
                    problem.solve(
                        solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True
                    )
                    expected = target if scale else np.zeros_like(target)
                    self.assertItemsAlmostEqual(variable.value, expected, places=5)
                    if cached is None:
                        cached = problem._cache.param_prog
                    self.assertIs(problem._cache.param_prog, cached)

    def test_products_of_parameters_remain_non_dpp(self):
        parameter = cp.Parameter((2, 2), value=np.eye(2))
        other = cp.Parameter((2, 2), value=np.ones((2, 2)))
        variable = cp.Variable((2, 2))
        expressions = [
            cp.kron(parameter, other),
            cp.kron(parameter, variable + other),
            cp.kron(cp.square(parameter), variable),
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertTrue(expression.is_dcp())
                self.assertFalse(expression.is_dcp(dpp=True))

    def test_affine_parameter_coefficients(self):
        values = np.arange(6.0).reshape(2, 3) + 1
        expected = np.array([[0.5, -2.0], [3.0, 1.0]])
        for backend, param_left in product(self.BACKENDS, [False, True]):
            with self.subTest(backend=backend, param_left=param_left):
                first = cp.Parameter(values.shape)
                second = cp.Parameter(values.shape)
                coefficient = 2 * first + second
                variable = cp.Variable(expected.shape)
                expression = (
                    cp.kron(coefficient, variable) if param_left
                    else cp.kron(variable, coefficient)
                )
                target = cp.Parameter(expression.shape)
                problem = cp.Problem(cp.Minimize(cp.sum_squares(expression - target)))
                self.assertTrue(problem.is_dpp())
                for scale in [1.0, -2.0]:
                    first.value = scale * values
                    second.value = np.ones_like(values)
                    coefficient_value = 2 * first.value + second.value
                    target.value = (
                        np.kron(coefficient_value, expected) if param_left
                        else np.kron(expected, coefficient_value)
                    )
                    problem.solve(
                        solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True
                    )
                    self.assertItemsAlmostEqual(variable.value, expected, places=5)

    def test_complex_parameters(self):
        values = np.array([[1 + 2j, 3 - 1j], [-2j, 2 + 1j]])
        expected = np.array([[0.5 - 1j, -2.0], [3j, 1 + 1j]])
        for backend, param_left in product(self.BACKENDS, [False, True]):
            with self.subTest(backend=backend, param_left=param_left):
                parameter = cp.Parameter(values.shape, complex=True, value=values)
                variable = cp.Variable(expected.shape, complex=True)
                expression = (
                    cp.kron(parameter, variable) if param_left
                    else cp.kron(variable, parameter)
                )
                target = (
                    np.kron(values, expected) if param_left
                    else np.kron(expected, values)
                )
                problem = cp.Problem(cp.Minimize(cp.sum_squares(expression - target)))
                self.assertTrue(problem.is_dpp())
                problem.solve(solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True)
                self.assertItemsAlmostEqual(variable.value, expected, places=5)

    def test_parameter_curvature(self):
        for sign, param_left, negate in product(
            [None, "nonneg", "nonpos"], [False, True], [False, True]
        ):
            with self.subTest(sign=sign, param_left=param_left, negate=negate):
                parameter = cp.Parameter((2, 1), **({sign: True} if sign else {}))
                operand = cp.square(cp.Variable((1, 2)))
                if negate:
                    operand = -operand
                expression = (
                    cp.kron(parameter, operand) if param_left
                    else cp.kron(operand, parameter)
                )
                expected = expression.is_convex(), expression.is_concave()
                with scopes.dpp_scope():
                    self.assertEqual(
                        (expression.is_convex(), expression.is_concave()), expected
                    )

    def test_nonlinear_parameterized_operands(self):
        for backend, param_left, positive in product(
            self.BACKENDS, [False, True], [False, True]
        ):
            with self.subTest(backend=backend, param_left=param_left, positive=positive):
                parameter = cp.Parameter((2, 1), nonneg=positive, nonpos=not positive)
                variable = cp.Variable((1, 2))
                operand = cp.square(variable)
                expression = (
                    cp.kron(parameter, operand) if param_left
                    else cp.kron(operand, parameter)
                )
                objective = (
                    cp.Minimize(cp.sum(expression) - 2 * cp.sum(variable)) if positive
                    else cp.Maximize(cp.sum(expression) + 2 * cp.sum(variable))
                )
                problem = cp.Problem(objective)
                self.assertTrue(problem.is_dpp())
                for scale in [1.0, 2.0]:
                    value = scale * np.array([[1.0], [2.0]])
                    parameter.value = value if positive else -value
                    problem.solve(
                        solver=cp.CLARABEL, canon_backend=backend, enforce_dpp=True
                    )
                    expected = np.ones(variable.shape) / np.sum(np.abs(parameter.value))
                    self.assertItemsAlmostEqual(variable.value, expected, places=5)
