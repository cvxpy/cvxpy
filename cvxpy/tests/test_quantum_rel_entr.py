import platform

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.atoms.affine.wraps import hermitian_wrap
from cvxpy.tests import solver_test_helpers as STH


def applychan(chan: np.array, rho: cp.Variable, rep: str, dim: tuple[int, int]):
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

def randH(n: int):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.conj().T)/2

def randRho(n: int):
    p = 10 * randH(n)
    p = (p @ p.conj().T)/np.trace(p @ p.conj().T)
    return p

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
        Nearest correlation matrix in the quantum relative entropy sense
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
        expect_obj = -36.19277
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
        % Quantum capacity of degradable channels

        % Example: amplitude damping channel
        % na = channel input dimension
        % nb = channel output dimension
        % ne = channel environment dimension
        % nf = degrading map environment dimension
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
        sth = TestQuantumRelEntr.make_test_1()
        sth.solve(**self.CLARABEL_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    @pytest.mark.skipif(not run_full_test_suite,\
                        reason="These tests are too slow to solve with CLARABEL")
    def test_2(self):
        sth = TestQuantumRelEntr.make_test_2()
        sth.solve(**self.MOSEK_ARGS)
        sth.verify_objective(places=2)
        sth.verify_primal_values(places=2)

    @pytest.mark.skipif(platform.system() == 'Linux' or not run_full_test_suite,\
                        reason="These tests are too slow to solve with CLARABEL")
    def test_3(self):
        sth = TestQuantumRelEntr.make_test_3()
        sth.solve(**self.MOSEK_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
