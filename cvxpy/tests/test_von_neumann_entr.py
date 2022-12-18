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
        """Expect un-specified EV to be 0.2"""
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
        """expect unspecified EV to be 0.4"""
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
    def make_test_3(quad_approx=False):
        A1 = np.array([[8.38972, 1.02671, 0.87991],
                       [1.02671, 8.41455, 7.31307],
                       [0.87991, 7.31307, 2.35915]])

        A2 = np.array([[6.92907, 4.37713, 5.11915],
                       [4.37713, 7.96725, 4.42217],
                       [5.11915, 4.42217, 2.72919]])

        b = np.array([19.16342, 17.62551])

        N = cp.Variable(shape=(3, 3), PSD=True)

        # The below problem generates the reference values:
        ref_X = cp.Variable(shape=(3, 3), diag=True)
        ref_objective = cp.Minimize(-cp.sum(cp.entr(cp.diag(ref_X))))
        ref_cons1 = trace(A1 @ ref_X) == b[0]
        ref_cons2 = trace(A2 @ ref_X) == b[1]
        ref_cons3 = ref_X >> 0
        ref_prob = cp.Problem(ref_objective, [ref_cons1, ref_cons2, ref_cons3])
        ref_obj_val = ref_prob.solve()
        ref_X_val = ref_X.value.A

        expect_N = ref_X_val
        if quad_approx:
            objective = cp.Minimize(-von_neumann_entr(N, (5, 5)))
        else:
            objective = cp.Minimize(-von_neumann_entr(N))
        obj_pair = (objective, ref_obj_val)
        cons1 = trace(A1 @ N) == b[0]
        cons2 = trace(A2 @ N) == b[1]
        cons3 = N - cp.diag(cp.diag(N)) == 0
        con_pairs = [
            (cons1, None),
            (cons2, None),
            (cons3, None),
        ]
        var_pairs = [
            (N, expect_N)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_3_exact(self):
        sth = self.make_test_3(quad_approx=False)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_3_approx(self):
        sth = self.make_test_3(quad_approx=True)
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_primal_feasibility(places=3)
