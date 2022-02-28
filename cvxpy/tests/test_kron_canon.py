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

    NOTE: this file is on the CVXPY 1.1.X branch. This class
    has the ability to test kron with a Variable in the left
    argument, but that feature was only introduced in CVXPY 1.2.
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
            prob.solve(solver='ECOS', abstol=1e-8, reltol=1e-8)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)
        with self.assertRaises(Exception):
            prob.solve(enforce_dpp=True)

    def test_gen_kronr_const(self):
        z_dims = (2, 2)
        for c_dims in TestKronRightVar.C_DIMS:
            Z, C, L, prob = self.make_kron_prob(z_dims, c_dims, param=False,
                                                var_left=False, seed=0)
            prob.solve(solver='ECOS', abstol=1e-8, reltol=1e-8)
            self.assertEqual(prob.status, cp.OPTIMAL)
            con_viols = prob.constraints[0].violation()
            self.assertLessEqual(np.max(con_viols), 1e-4)
            self.assertItemsAlmostEqual(Z.value, L, places=4)
