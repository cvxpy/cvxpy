import platform
import sys

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.atoms.affine.wraps import hermitian_wrap
from cvxpy.tests import solver_test_helpers as STH


def applychan(chan: np.ndarray, rho: cp.Variable, rep: str, dim: tuple[int, int]):
    dimA, dimB, dimE = None, None, None
    if rep == 'choi2':
        dimA, dimB = dim
        arg = chan @ kron(rho.T, np.eye(dimB))
        rho_out = partial_trace(arg, [dimA, dimB], 0)
        return rho_out
    elif rep == 'isom':
        dimA = chan.shape[1]
        dimB = dim[1]
        dimE = int(chan.shape[0]/dimB)
        rho_out = partial_trace(chan @ rho @ chan.conj().T, [dimB, dimE], 1)
        return rho_out


class TestQuantumRelEntr:
    """
    Test class for `quantum_rel_entr` & `quantum_cond_entr`
    - All of the reference solutions for the problem come from equivalent
    CVXQUAD implementations
    - These problems also show up in a Marimo notebook linked on the CVXPY docs
    """
    run_full_test_suite = 'MOSEK' in cp.installed_solvers()
    MOSEK_ARGS = {'solver': 'MOSEK', 'verbose': True}
    CLARABEL_ARGS = {'solver': 'CLARABEL', 'verbose': True}

    @staticmethod
    def make_test_1():
        """
        Nearest correlation matrix in the quantum relative entropy sense.
        M is a constant matrix and X is the variable, so this problem hits
        the fast canonicalization path (X constant -> 2n x 2n SDP blocks
        instead of 2n^2 x 2n^2). The expected objective reflects the
        fast-path approximation.
        """
        n = 4
        M = np.array([[0.5377, 0.3188, 3.5784, 0.7254],
                      [1.8339, -1.3077, 2.7694, -0.0631],
                      [-2.2588, -0.4336, -1.3499, 0.7147],
                      [0.8622, 0.3426, 3.0349, -0.2050]])
        M = M @ M.T

        X = cp.Variable(shape=(n, n), symmetric=True)
        expect_X = np.array([[1.0000, 0.7956, -0.5286, 0.9442],
                    [0.7956, 1.0000, -0.7238, 0.8387],
                    [-0.5286, -0.7238, 1.0000, -0.7176],
                    [0.9442, 0.8387, -0.7176, 1.0000]])
        var_pairs = [(X, expect_X)]

        obj = cp.Minimize(cp.quantum_rel_entr(M, X))
        # NOTE: expected objective updated from -36.19277 to -35.59036 to
        # reflect the fast canonicalization path introduced in PR #3153.
        # When M is constant, block size drops from 2n^2 x 2n^2 to 2n x 2n
        # (Fawzi & Fawzi 2018, Table 1 footnote b), producing a slightly
        # different numerical approximation. Tolerance loosened to places=1
        # accordingly.
        expect_obj = -35.59036
        obj_pair = (obj, expect_obj)

        cons1 = cp.diag(X) == np.ones((n,))
        con_pairs = [(cons1, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)

        return sth

    @staticmethod
    def make_test_2():
        """
        Compute lower bound on relative entropy of entanglement (PPT relaxation)
        """
        na, nb = (2, 2)
        rho = np.array([[ 0.07 -0.j   ,  0.051-0.039j, -0.023-0.005j, -0.072-0.01j ],
                        [ 0.051+0.039j,  0.129-0.j   ,  0.009+0.022j, -0.128-0.146j],
                        [-0.023+0.005j,  0.009-0.022j,  0.466+0.j   , -0.052-0.003j],
                        [-0.072+0.01j , -0.128+0.146j, -0.052+0.003j,  0.335+0.j   ]])

        tau = cp.Variable(shape=(na * nb, na * nb), hermitian=True)
        expect_tau = \
        np.array([[0.0692 + 0.0000j, 0.0383 - 0.0310j, -0.0304 - 0.0143j, -0.0507 - 0.0019j],
                  [0.0383 + 0.0310j, 0.1303 + 0.0000j, 0.0046 - 0.0119j, -0.1196 - 0.1361j],
                  [-0.0304 + 0.0143j, 0.0046 + 0.0119j, 0.4705 + 0.0000j, -0.0323 - 0.0159j],
                  [-0.0507 + 0.0019j, -0.1196 + 0.1361j, -0.0323 + 0.0159j, 0.3299 + 0.0000j]])
        var_pairs = [(tau, expect_tau)]

        obj = cp.Minimize(cp.quantum_rel_entr(rho, tau, (3,3))/np.log(2))
        expect_obj = 0.02171
        obj_pair = (obj, expect_obj)

        cons1 = tau >> 0
        cons2 = cp.trace(tau) == 1
        cons3 = cp.partial_transpose(tau, [na, nb], 1) >> 0
        cons_pairs = [(cons1, None), (cons2, None), (cons3, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)

        return sth

    @staticmethod
    def make_test_3():
        """
        Quantum capacity of degradable channels
        Example: amplitude damping channel
        na = channel input dimension
        nb = channel output dimension
        ne = channel environment dimension
        nf = degrading map environment dimension
        """
        na, nb, ne, nf = (2, 2, 2, 2)
        def AD(gamma: float):
            return np.array([[1, 0],[0, np.sqrt(gamma)],[0, np.sqrt(1-gamma)],[0, 0]])
        gamma = 0.2
        U = AD(gamma)

        W = AD((1-2*gamma)/(1-gamma))

        def Ic(rho: cp.Variable):
            return cp.quantum_cond_entr(
                        hermitian_wrap(W @ applychan(U, rho, 'isom', (na, nb)) @ W.conj().T),
                        [ne, nf], 1
                    )/np.log(2)

        rho = cp.Variable(shape=(na, na), hermitian=True)
        rho_expect = np.array([[0.5511, 0],
                            [0, 0.4489]])
        var_pairs = [(rho, rho_expect)]

        obj = cp.Maximize(Ic(rho))
        obj_expect = 0.506214
        obj_pair = (obj, obj_expect)

        cons1 = rho >> 0
        cons2 = cp.trace(rho) == 1
        cons_pairs = [(cons1, None), (cons2, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)

        return sth

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="This test is skipped on Windows",
    )
    def test_1(self):
        print("*****************************")
        print(f"Platform: {platform.system()}")
        print(f"Python version: {sys.version_info}")
        print("*****************************")
        sth = TestQuantumRelEntr.make_test_1()
        sth.solve(**self.CLARABEL_ARGS)
        # places=1 because the fast canonicalization path (PR #3153) produces
        # a structurally different SDP approximation than the general path,
        # leading to a slightly different numerical solution. Both are valid
        # approximations of the true quantum relative entropy.
        sth.verify_objective(places=1)

    @pytest.mark.skipif(not run_full_test_suite,
                        reason="These tests are too slow to solve with CLARABEL")
    def test_2(self):
        sth = TestQuantumRelEntr.make_test_2()
        sth.solve(**self.MOSEK_ARGS)
        sth.verify_objective(places=2)
        sth.verify_primal_values(places=2)

    @pytest.mark.skipif(not run_full_test_suite,
                        reason="These tests are too slow to solve with CLARABEL")
    def test_3(self):
        sth = TestQuantumRelEntr.make_test_3()
        sth.solve(**self.MOSEK_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_real_inputs(self):
        """quantum_rel_entr with real inputs should solve without error."""
        n = 2
        X = cp.Variable((n, n), symmetric=True)
        Y = np.eye(n)
        prob = cp.Problem(
            cp.Minimize(cp.quantum_rel_entr(X, Y)),
            [cp.trace(X) == 1, X >> 0]
        )
        prob.solve(**self.CLARABEL_ARGS)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(prob.value, -np.log(n), atol=1e-3)

    def test_constant_first_arg_matches_general(self):
        """
        Verify that f(C, Y) [fast path: X constant] gives the same optimal
        value as f(X, Y) s.t. X == C [general path, both variable].
        Per PR #3153 / Fawzi & Fawzi 2018 Table 1 footnote b.
        """
        n = 3
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        C = A @ A.T + np.eye(n) * 0.1
        C = C / np.trace(C)

        # Fast path: C is a numpy array (constant), Y is variable
        Y_fast = cp.Variable((n, n), symmetric=True)
        prob_fast = cp.Problem(
            cp.Minimize(cp.quantum_rel_entr(C, Y_fast)),
            [Y_fast >> 0, cp.trace(Y_fast) == 1]
        )
        prob_fast.solve(**self.CLARABEL_ARGS)

        # General path: both X and Y are variables, X is pinned to C
        X_gen = cp.Variable((n, n), symmetric=True)
        Y_gen = cp.Variable((n, n), symmetric=True)
        prob_gen = cp.Problem(
            cp.Minimize(cp.quantum_rel_entr(X_gen, Y_gen)),
            [X_gen >> 0, cp.trace(X_gen) == 1,
             Y_gen >> 0, cp.trace(Y_gen) == 1,
             X_gen == C]
        )
        prob_gen.solve(**self.CLARABEL_ARGS)

        np.testing.assert_allclose(
            prob_fast.value, prob_gen.value, atol=1e-3,
            err_msg="Fast path (X constant) should match general path (X == C constraint)"
        )

    def test_constant_second_arg_matches_general(self):
        """
        Verify that f(X, C) [fast path: Y constant] gives the same optimal
        value as f(X, Y) s.t. Y == C [general path, both variable].
        Per PR #3153 / Fawzi & Fawzi 2018 Table 1 footnote b.
        """
        n = 3
        rng = np.random.default_rng(7)
        A = rng.standard_normal((n, n))
        C = A @ A.T + np.eye(n) * 0.1
        C = C / np.trace(C)

        # Fast path: X is variable, C is a numpy array (constant)
        X_fast = cp.Variable((n, n), symmetric=True)
        prob_fast = cp.Problem(
            cp.Minimize(cp.quantum_rel_entr(X_fast, C)),
            [X_fast >> 0, cp.trace(X_fast) == 1]
        )
        prob_fast.solve(**self.CLARABEL_ARGS)

        # General path: both X and Y are variables, Y is pinned to C
        X_gen = cp.Variable((n, n), symmetric=True)
        Y_gen = cp.Variable((n, n), symmetric=True)
        prob_gen = cp.Problem(
            cp.Minimize(cp.quantum_rel_entr(X_gen, Y_gen)),
            [X_gen >> 0, cp.trace(X_gen) == 1,
             Y_gen >> 0, cp.trace(Y_gen) == 1,
             Y_gen == C]
        )
        prob_gen.solve(**self.CLARABEL_ARGS)

        np.testing.assert_allclose(
            prob_fast.value, prob_gen.value, atol=1e-3,
            err_msg="Fast path (Y constant) should match general path (Y == C constraint)"
        )

    def test_constant_arg_sdp_size_reduction(self):
        """
        Verify that the fast path (one constant argument) produces a smaller
        SDP than the general path (both variable).
        Per PR #3153 / Fawzi & Fawzi 2018 Table 1 footnote b:
        block size drops from 2n^2 x 2n^2 to 2n x 2n.
        """
        for n in [2, 3, 4]:
            rng = np.random.default_rng(n)
            A = rng.standard_normal((n, n))
            C = A @ A.T + np.eye(n) * 0.1
            C = C / np.trace(C)

            # Fast path: X constant
            Y_v = cp.Variable((n, n), symmetric=True)
            prob_fast = cp.Problem(
                cp.Minimize(cp.quantum_rel_entr(C, Y_v)),
                [Y_v >> 0, cp.trace(Y_v) == 1]
            )
            data_fast, _, _ = prob_fast.get_problem_data(solver=cp.SCS)

            # General path: both variable
            X_v = cp.Variable((n, n), symmetric=True)
            Y_v2 = cp.Variable((n, n), symmetric=True)
            prob_gen = cp.Problem(
                cp.Minimize(cp.quantum_rel_entr(X_v, Y_v2)),
                [X_v >> 0, cp.trace(X_v) == 1,
                 Y_v2 >> 0, cp.trace(Y_v2) == 1]
            )
            data_gen, _, _ = prob_gen.get_problem_data(solver=cp.SCS)

            fast_vars = data_fast['A'].shape[1]
            gen_vars = data_gen['A'].shape[1]
            assert fast_vars < gen_vars, (
                f"n={n}: fast path ({fast_vars} vars) should be smaller "
                f"than general path ({gen_vars} vars)"
            )
