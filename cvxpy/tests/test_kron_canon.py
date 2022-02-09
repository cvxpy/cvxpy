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

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestKron(BaseTest):

    def base_test_kron_canon_1(self, PARAM) -> None:
        """Test canonicalization of kron with a variable as
        the first argument, by using it in optimization problems.

        kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
                     [M[1,0] * N   , ..., M[1, end] * N  ]
                         ...
                         [M[end, 0] * N, ..., M[end, end] * N]
        """
        X = cp.Variable(shape=(2, 2), symmetric=True)
        b_val = 1.5 * np.ones((1, 1))
        if PARAM:
            b = cp.Parameter(shape=(1, 1))
            b.value = b_val
        else:
            b = cp.Constant(b_val)
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        kronX = cp.kron(X, b)  # should be equal to X

        objective = cp.Minimize(cp.sum(X.flatten()))
        constraints = [U >= kronX, kronX >= L]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.assertItemsAlmostEqual(X.value, np.array([[0.5, 2], [2, 3]]) / 1.5)
        objective = cp.Maximize(cp.sum(X.flatten()))
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.assertItemsAlmostEqual(X.value, np.array([[10, 11], [11, 13]]) / 1.5)
        pass

    def test_kron_canon_1_param(self):
        self.base_test_kron_canon_1(PARAM=True)

    def test_kron_canon_1(self):
        self.base_test_kron_canon_1(PARAM=False)

    def base_test_kron_canon_2(self, PARAM) -> None:
        y = cp.Variable(shape=(1, 1))
        A_val = np.array([[1., 2.], [3., 4.]])
        L = np.array([[0.5, 1], [2, 3]])
        U = np.array([[10, 11], [12, 13]])
        if PARAM:
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

    def test_kron_canon_2_param(self):
        self.base_test_kron_canon_2(PARAM=True)

    def test_kron_canon_2(self):
        self.base_test_kron_canon_2(PARAM=False)

    def base_test_kron_canon_3(self, MIN, PARAM) -> None:
        for c_dims in [(1, 1)]:
            print(c_dims)
            Z = cp.Variable(shape=(2, 3))
            np.random.seed(0)
            C_value = np.random.rand(*c_dims)
            if PARAM:
                C = cp.Parameter(shape=c_dims)
                C.value = C_value
            else:
                C = cp.Constant(C_value)
            L = np.random.rand(*Z.shape)
            U = L + np.random.rand(*Z.shape)
            kron = cp.kron(Z, C)
            constraints = [cp.kron(U, C) >= kron, kron >= cp.kron(L, C)]
            obj_expr = cp.sum(Z)

            if MIN:
                prob = cp.Problem(cp.Minimize(obj_expr), constraints)
                prob.solve()
                Z_actual = Z.value
                Z_expect = L
                self.assertItemsAlmostEqual(Z_actual, Z_expect)
            else:
                prob = cp.Problem(cp.Maximize(obj_expr), constraints)
                prob.solve()
                Z_actual = Z.value
                Z_expect = U
                self.assertItemsAlmostEqual(Z_actual, Z_expect)
        pass

    def test_kron_canon_3_min(self):
        self.base_test_kron_canon_3(MIN=True, PARAM=False)

    def test_kron_canon_3_max(self):
        self.base_test_kron_canon_3(MIN=False, PARAM=False)

    def test_kron_canon_3_min_param(self):
        self.base_test_kron_canon_3(MIN=True, PARAM=True)

    def test_kron_canon_3_max_param(self):
        self.base_test_kron_canon_3(MIN=False, PARAM=True)
