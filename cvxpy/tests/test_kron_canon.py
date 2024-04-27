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
from typing import Tuple

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestKron(BaseTest):
    """
    The Kronecker product of matrices M, N is :

        kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
                     [M[1,0] * N   , ..., M[1, end] * N  ]
                     ...
                     [M[end, 0] * N, ..., M[end, end] * N]
    """

    @staticmethod
    def make_kron_prob(z_dims: Tuple[int],
                       c_dims: Tuple[int],
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
            # The cvxcore function get_kronl_mat doesn't work when C is a Parameter.
            # We get around this by having kron be non-dpp, but this comes at
            # the price of eliminating the speed benefit of using Parameter objects.
            # We'll eventually need to extend get_kronl_mat so that it supports
            # Parameters. Until then, I'll make a note that tests here DO PASS
            # with the existing get_kronl_mat implementation if we use the following
            # constraints: [cp.kron(Z - L, C) >= 0, Z >= 0].
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
