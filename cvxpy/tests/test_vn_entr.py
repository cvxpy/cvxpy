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

import cvxpy as cp
from cvxpy.atoms import vn_entr
from cvxpy.tests import solver_test_helpers as STH


class Test_vn_entr:
    @staticmethod
    def make_test_1():
        """(2,2) matrix, 100 largest ev, 1e-3 off-diagonal element"""
        n = 2
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[0.36787973, 0.00099987],
                             [0.00099987, 0.36787973]])
        eps = 1e2
        cons1 = N >> 0
        cons2 = N << eps*np.eye(N.shape[0])
        cons3 = N[1][0] == 1e-3
        objective = cp.Maximize(vn_entr(N))
        obj_pair = (objective, 0.735756164739688)
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_1(self):
        sth = Test_vn_entr.make_test_1()
        sth.solve(solver='SCS')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_2():
        """(3,3) matrix, 1000 largest ev, 1e-2 off-diagonal element"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[3.66497984e-01, -9.66999473e-14, 9.99852746e-03],
                             [-9.66999473e-14, 3.70615053e-01, -3.41058370e-13],
                             [9.99852746e-03, -3.41058370e-13, 3.66497984e-01]])
        eps = 1e3
        cons1 = N >> 0
        cons2 = N << eps*np.eye(N.shape[0])
        cons3 = N[2][0] == 1e-2
        objective = cp.Maximize(vn_entr(N))
        obj_pair = (objective, 1.103350176968849)
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_2(self):
        sth = Test_vn_entr.make_test_2()
        sth.solve(solver='SCS')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def make_test_3():
        """(3,3) matrix, 1000 largest ev, 1e-2 off diagonal element under a unitary transform"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[3.70615097e-01, -6.50164940e-15, -2.51416162e-13],
                             [-6.50164940e-15, 3.66498027e-01, -9.99852741e-03],
                             [-2.51416162e-13, -9.99852741e-03, 3.66498027e-01]])
        U = cp.Constant(np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
        eps = 1e3
        cons1 = N >> 0
        cons2 = N << eps*np.eye(N.shape[0])
        cons3 = N[2][0] == 1e-2
        N = U.T@N@U
        objective = cp.Maximize(vn_entr(N))
        obj_pair = (objective, 1.1033501770078251)
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_3(self):
        sth = Test_vn_entr.make_test_3()
        sth.solve(solver='SCS')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_4():
        """(3,3) matrix, [1,0,1] EV, 3 ev, 1e-5 off-diagonal value"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[2.99999997e+00, -1.59980927e-15, 1.81681520e-08],
                             [-1.59980927e-15, 2.99990732e+00, 1.61179597e-15],
                             [1.81681520e-08, 1.61179597e-15, 2.99999997e+00]])
        cons1 = N >> 0
        v = cp.Constant(np.array([1, 0, 1]))
        mu = 3
        cons2 = N @ v == mu * v
        cons3 = N[2][0] == 1e-5
        objective = cp.Maximize(vn_entr(N))
        obj_pair = (objective, -9.887315950231013)
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_4(self):
        sth = Test_vn_entr.make_test_4()
        sth.solve(solver='SCS')
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
