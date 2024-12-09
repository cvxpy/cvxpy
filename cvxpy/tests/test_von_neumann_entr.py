"""
Copyright, 2022 - the CVXPY authors

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
import scipy as sp

import cvxpy as cp
from cvxpy import trace
from cvxpy.atoms import von_neumann_entr
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.utilities.linalg import onb_for_orthogonal_complement


class Test_von_neumann_entr:

    if 'MOSEK' in cp.installed_solvers():
        SOLVE_ARGS = {'solver': 'MOSEK', 'verbose': True}
    else:
        SOLVE_ARGS = {'solver': 'SCS', 'eps': 1e-6, 'max_iters': 500_000,
                      'verbose': True}

    @staticmethod
    def make_test_1(complex):
        """Enforce an upper bound of 0.8 on trace(N);
        Expect N's unspecified eigenvalue to be 0.2"""
        n = 3
        if hasattr(np.random, 'default_rng'):
            rng = np.random.default_rng(0)
        else:
            rng = np.random.RandomState(0)
        if complex:
            N = cp.Variable(shape=(n, n), hermitian=True)
            V12 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        else:
            N = cp.Variable(shape=(n, n), PSD=True)
            V12 = rng.normal(size=(n, n))
        V12 = sp.linalg.qr(V12)[0][:, :2]
        mu12 = np.array([0.5, 0.1])
        trace_bound = 0.8
        cons1 = N @ V12 == V12 * mu12
        cons2 = trace(N) <= trace_bound
        objective = cp.Maximize(von_neumann_entr(N))

        V3 = onb_for_orthogonal_complement(V12).reshape((n, 1))
        mu3 = trace_bound - np.sum(mu12)
        expect_mu = np.concatenate([mu12, [mu3]])
        expect_V = np.column_stack([V12, V3])
        if complex:
            expect_N = (expect_V * expect_mu) @ expect_V.conj().T
        else:
            expect_N = (expect_V * expect_mu) @ expect_V.T
        expect_obj = cp.sum(cp.entr(expect_mu)).value

        obj_pair = (objective, expect_obj)
        con_pairs = [(cons1, None), (cons2, None)]
        var_pairs = [(N, expect_N)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_1_real(self):
        sth = Test_von_neumann_entr.make_test_1(False)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_1_complex(self):
        sth = Test_von_neumann_entr.make_test_1(True)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def make_test_2(quad_approx):
        """Enforce a lower bound of 0.9 on trace(N);
        Expect N's unspecified eigenvalue to be 0.4"""
        n = 3
        N = cp.Variable(shape=(n, n), PSD=True)
        V12 = np.array([[-0.12309149, 0.90453403],
                        [-0.49236596, 0.30151134],
                        [-0.86164044, -0.30151134]])
        mu12 = np.array([0.3, 0.2])
        trMin = 0.9
        cons1 = N @ V12 == V12 * mu12
        cons2 = trace(N) >= trMin
        if quad_approx:
            objective = cp.Maximize(von_neumann_entr(N, (5, 5)))
        else:
            objective = cp.Maximize(von_neumann_entr(N))

        V3 = onb_for_orthogonal_complement(V12).reshape((n, 1))
        mu3 = trMin - np.sum(mu12)
        expect_mu = np.concatenate([mu12, [mu3]])
        expect_V = np.column_stack([V12, V3])
        expect_N = (expect_V * expect_mu) @ expect_V.T
        expect_obj = cp.sum(cp.entr(expect_mu)).value

        obj_pair = (objective, expect_obj)
        con_pairs = [(cons1, None), (cons2, None)]
        var_pairs = [(N, expect_N)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_2_exact(self):
        sth = Test_von_neumann_entr.make_test_2(False)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_2_approx(self):
        sth = Test_von_neumann_entr.make_test_2(True)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @staticmethod
    def sum_entr_approx(a: cp.Expression, apx_m: int, apx_k: int):
        n = a.size
        epi_vec = cp.Variable(shape=n)
        b = cp.Constant(np.ones(n))
        con = cp.constraints.RelEntrConeQuad(a, b, epi_vec, apx_m, apx_k)
        objective = cp.Minimize(cp.sum(epi_vec))
        return objective, con

    @staticmethod
    def make_test_3(quad_approx=False, real=False):
        np.random.seed(0)
        ###################################################
        #
        #   Construct matrix/vector coefficient data
        #
        ###################################################
        apx_m, apx_k = 2, 2
        A1_real = np.array([[8.38972, 1.02671, 0.87991],
                            [1.02671, 8.41455, 7.31307],
                            [0.87991, 7.31307, 2.35915]])

        A2_real = np.array([[6.92907, 4.37713, 5.11915],
                            [4.37713, 7.96725, 4.42217],
                            [5.11915, 4.42217, 2.72919]])
        if real:
            U = np.eye(3)
            A1 = A1_real
            A2 = A2_real
        else:
            randmat = 1j * np.random.normal(size=(3, 3))
            randmat += np.random.normal(size=(3, 3))
            U = sp.linalg.qr(randmat)[0]
            A1 = U @ A1_real @ U.conj().T
            A2 = U @ A2_real @ U.conj().T
        b = np.array([19.16342, 17.62551])

        ###################################################
        #
        #   define and solve a reference problem
        #
        ###################################################
        diag_X = cp.Variable(shape=(3,))
        ref_X = cp.diag(diag_X)
        if real:
            ref_cons = [trace(A1 @ ref_X) == b[0],
                        trace(A2 @ ref_X) == b[1]]
        else:
            conjugated_X = U @ ref_X @ U.conj().T
            ref_cons = [trace(A1 @ conjugated_X) == b[0],
                        trace(A2 @ conjugated_X) == b[1]]
        if quad_approx:
            ref_objective, con = Test_von_neumann_entr.sum_entr_approx(diag_X, apx_m, apx_k)
            ref_cons.append(con)
        else:
            ref_objective = cp.Minimize(-cp.sum(cp.entr(diag_X)))
        ref_prob = cp.Problem(ref_objective, ref_cons)
        ref_obj_val = ref_prob.solve()

        ###################################################
        #
        #   define a new problem that is equivalent to the
        #   reference, but makes use of von_neumann_entr.
        #
        ###################################################
        if real:
            N = cp.Variable(shape=(3, 3), PSD=True)
            cons = [trace(A1 @ N) == b[0],
                    trace(A2 @ N) == b[1],
                    N - cp.diag(cp.diag(N)) == 0]
            expect_N = ref_X.value
        else:
            N = cp.Variable(shape=(3, 3), hermitian=True)
            aconj_N = U.conj().T @ N @ U
            cons = [trace(A1 @ N) == b[0],
                    trace(A2 @ N) == b[1],
                    aconj_N - cp.diag(cp.diag(aconj_N)) == 0]
            expect_N = conjugated_X.value
        if quad_approx:
            objective = cp.Minimize(-von_neumann_entr(N, (apx_m, apx_k)))
        else:
            objective = cp.Minimize(-von_neumann_entr(N))

        ###################################################
        #
        #   construct and return the SolverTestHelper
        #
        ###################################################
        obj_pair = (objective, ref_obj_val)
        var_pairs = [(N, expect_N)]
        con_pairs = [(con, None) for con in cons]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_3_exact_real(self):
        sth = self.make_test_3(quad_approx=False, real=True)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_3_approx_real(self):
        sth = self.make_test_3(quad_approx=True, real=True)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_primal_feasibility(places=3)

    def test_3_exact_complex(self):
        sth = self.make_test_3(quad_approx=False, real=False)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_3_approx_complex(self):
        sth = self.make_test_3(quad_approx=True, real=False)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_primal_feasibility(places=3)

    @staticmethod
    def make_test_4():
        """
        Compute capacity of a cq-channel
        """
        rho1 = np.array([[1, 0],
                        [0, 0]])
        rho2 = 0.5 * np.ones((2, 2))
        H1 = cp.von_neumann_entr(rho1)
        H2 = cp.von_neumann_entr(rho2)

        p1 = cp.Variable()
        p2 = cp.Variable()
        p1_expect = 0.5
        p2_expect = 0.5
        var_pairs = [(p1, p1_expect), (p2, p2_expect)]

        obj = cp.Maximize((cp.von_neumann_entr(p1 * rho1 + p2 * rho2) - p1 * H1 - \
                           p2 * H2)/np.log(2))
        obj_expect = 0.60088
        obj_pair = (obj, obj_expect)

        cons1 = p1 >= 0
        cons2 = p2 >= 0
        cons3 = p1 + p2 == 1
        cons_pair = [(cons1, None), (cons2, None), (cons3, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, cons_pair)

        return sth

    def test_4(self):
        sth = Test_von_neumann_entr.make_test_4()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
