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
from numpy.linalg import LinAlgError

import cvxpy as cp
from cvxpy import trace
from scipy.linalg import logm
from cvxpy.atoms import von_neumann_entr
from cvxpy.tests import solver_test_helpers as STH


class Test_von_neumann_entr:

    if 'MOSEK' in cp.installed_solvers():
        SOLVE_ARGS = {'solver': 'MOSEK', 'verbose': True}
    else:
        SOLVE_ARGS = {'solver': 'SCS', 'eps': 1e-6, 'max_iters': 500_000,
                      'verbose': True}

    # @staticmethod
    # def projector(A):
    #     """
    #     Helper function for computing $n-th$ eigenvector, given the other $(n-1)$
    #     A: matrix with dimensions (n, n-1), n-1 eigenvectors arranged column-wise
    #     """
    #     # Generate an orthogonal basis for the (n-1) dimensional space
    #     Q = np.linalg.qr(A)[0]
    #     # grab the projector for this space
    #     P = Q @ Q.T
    #     # grab the projector for it's orthogonal space
    #     P_orth = np.identity(Q.shape[0]) - P
    #     x = np.random.rand(Q.shape[0])
    #     # take the projection of some arbitrary vector in R^{n} onto this space to get v_{n}
    #     v_n = P_orth @ x / np.linalg.norm(P_orth @ x)
    #     return np.resize(v_n, (Q.shape[0],1))

    @staticmethod
    def make_test_1():
        """Expect un-specified EV to be 0.2"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[0.14523781, 0.02381009, 0.10238067],
                             [0.02381009, 0.28095125, 0.13809581],
                             [0.10238067, 0.13809581, 0.37380924]])
        v1 = np.array([[-0.26726124],
                       [-0.53452248],
                       [-0.80178373]])
        v2 = np.array([[ 0.87287156],
                       [ 0.21821789],
                       [-0.43643578]])
        lambda1 = 0.5
        lambda2 = 0.1
        trMax = 0.8
        cons1 = N @ v1 == lambda1 * v1
        cons2 = N @ v2 == lambda2 * v2
        cons3 = trace(N) <= trMax
        objective = cp.Maximize(von_neumann_entr(N))
        obj_pair = (objective, 0.8987186478352693)
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
        sth = Test_von_neumann_entr.make_test_1()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_2():
        """expect unspecified EV to be 0.4"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[ 0.23484857, -0.06060623,  0.04393948],
                             [-0.06060623,  0.3575761 , -0.0242426 ],
                             [ 0.04393948, -0.0242426 ,  0.30757584]])
        v1 = np.array([[-0.12309149],
                       [-0.49236596],
                       [-0.86164044]])
        v2 = np.array([[ 0.90453403],
                       [ 0.30151134],
                       [-0.30151134]])
        lambda1 = 0.3
        lambda2 = 0.2
        trMax = 0.9
        cons1 = N @ v1 == lambda1 * v1
        cons2 = N @ v2 == lambda2 * v2
        cons3 = trace(N) >= trMax
        objective = cp.Maximize(von_neumann_entr(N))
        obj_pair = (objective, 1.049595673951)
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
        sth = Test_von_neumann_entr.make_test_2()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def make_test_3():
        """Expect unspecified EV to be 0.35"""
        n = 4
        N = cp.Variable(shape=(n, n), PSD=True)
        expect_N = np.array([[ 0.15567716, -0.05194798, -0.04646885,  0.05940634],
                             [-0.05194798,  0.18639235,  0.01639256, -0.01750361],
                             [-0.04646885,  0.01639256,  0.25662141, -0.07654513],
                             [ 0.05940634, -0.01750361, -0.07654513,  0.20130907]])
        v1 = np.array([[-0.18257419],
                    [-0.36514837],
                    [-0.54772256],
                    [-0.73029674]])
        v2 = np.array([[-8.16496581e-01],
                    [-4.08248290e-01],
                    [ 2.22044605e-16],
                    [ 4.08248290e-01]])
        v3 = np.array([[-0.37407225],
                    [ 0.79697056],
                    [-0.47172438],
                    [ 0.04882607]])
        lambda1 = 0.15
        lambda2 = 0.1
        lambda3 = 0.2
        trMax = 0.8
        cons1 = N @ v1 == lambda1 * v1
        cons2 = N @ v2 == lambda2 * v2
        cons3 = N @ v3 == lambda3 * v3
        cons4 = trace(N) <= trMax
        objective = cp.Maximize(von_neumann_entr(N))
        obj_pair = (objective, 1.2041518326298097)
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None),
            (cons4, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_3(self):
        sth = Test_von_neumann_entr.make_test_3()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_4():
        m = 4
        A1 = np.array([[8.38972 ,1.02671 ,0.87991],
                       [1.02671 ,8.41455 ,7.31307],
                       [0.87991 ,7.31307 ,2.35915]])

        A2 = np.array([[6.92907 ,4.37713 ,5.11915],
                       [4.37713 ,7.96725 ,4.42217],
                       [5.11915 ,4.42217 ,2.72919]])

        A3 = np.array([[1.59465 ,6.97858 ,1.19771],
                       [6.97858 ,1.95050 ,1.16576],
                       [1.19771 ,1.16576 ,7.04837]])

        A4 = np.array([[-3,  0,  0],
                       [ 0, -2,  0],
                       [ 0,  0, -1]])

        X_ = np.array([[42, 0, 0],
                       [0, 20, 0],
                       [0, 0, 69]])

        A = [A1, A2, A3, A4]
        y_ = np.array([[0.56625887],
                       [0.82940523],
                       [0.92765576],
                       [16]])

        running = np.zeros(A[0].shape)
        for index in range(m):
            running += A[index] * y_[index]
        C = logm(X_) + np.eye(3)
        S = C - running
        try:
            tmp = np.linalg.cholesky(S)
        except LinAlgError:
            print("If S is not PSD, then the above test case is invalid")

        b = np.array([np.trace(A[index] @ X_) for index in range(m)]).reshape((4, 1))

        N = cp.Variable(shape = (3,3), PSD = True)
        expect_N = np.array([[ 4.20000004e+01, -1.02890475e-08,  1.11583350e-08],
                             [-1.02890475e-08,  1.99999995e+01,  1.14357171e-08],
                             [ 1.11583350e-08,  1.14357171e-08,  6.90000002e+01]])
        objective = cp.Minimize(-von_neumann_entr(N))
        obj_pair = (objective, 509.05011909369927)
        cons1 = trace(A1 @ N) == b[0]
        cons2 = trace(A2 @ N) == b[1]
        cons3 = trace(A3 @ N) == b[2]
        cons4 = trace(A4 @ N) == b[3]
        cons5 = N - cp.diag(cp.diag(N)) == 0
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None),
            (cons4, None),
            (cons5, None)
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_4(self):
        sth = Test_von_neumann_entr.make_test_4()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
