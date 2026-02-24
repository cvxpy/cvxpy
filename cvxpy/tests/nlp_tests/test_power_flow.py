import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestPowerFlowIPOPT:
    """Power flow problem from DNLP paper"""

    def test_power_flow_dense_formulation(self):
        # -----------------------------------------------------------------------------------
        #                             Define prob data
        # -----------------------------------------------------------------------------------
        N = 9
        p_min = np.zeros(N)
        p_max = np.zeros(N)
        q_min = np.zeros(N)
        q_max = np.zeros(N)
        p_min[[0, 1, 2]] = [10, 10, 10]
        p_max[[0, 1, 2]] = [250, 300, 270]
        q_min[[0, 1, 2]] = [-5, -5, -5]
        p_min[[4,6,8]] = p_max[[4,6,8]] = [-54, -60, -75]
        q_min[[4,6,8]] = q_max[[4,6,8]] = [-18, -21, -30]
        v_min, v_max = 0.9, 1.1
        
        # -----------------------------------------------------------------------------------
        #                     Define admittance matrices
        # -----------------------------------------------------------------------------------
        # Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
        branch_data = np.array([
            [0, 3, 0.0, 0.0576, 0.0],
            [3, 4, 0.017, 0.092, 0.158],
            [5, 4, 0.039, 0.17, 0.358],
            [2, 5, 0.0, 0.0586, 0.0],
            [5, 6, 0.0119, 0.1008, 0.209],
            [7, 6, 0.0085, 0.072, 0.149],
            [1, 7, 0.0, 0.0625, 0.0],
            [7, 8, 0.032, 0.161, 0.306],
            [3, 8, 0.01, 0.085, 0.176],
        ])

        M = branch_data.shape[0]  # Number of branches
        base_MVA = 100

        # Build incidence matrix A
        from_bus = branch_data[:, 0].astype(int)
        to_bus = branch_data[:, 1].astype(int)
        A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
            csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))

        # Network impedance
        z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA

        # Bus admittance matrix Y_0
        Y_0 = A @ diags(1.0 / z) @ A.T

        # Shunt admittance from line charging
        y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
        Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
        Y_sh = diags(Y_sh_diag)

        # Extract conductance and susceptance matrices
        G0 = np.real(Y_0.toarray())  # Conductance matrix
        B0 = np.imag(Y_0.toarray())  # Susceptance matrix
        G_sh = np.real(Y_sh.toarray())  # Shunt conductance
        B_sh = np.imag(Y_sh.toarray())  #
        G = G0 + G_sh
        B = B0 + B_sh
        

        # -----------------------------------------------------------------------------------
        #                         Define optimization prob
        # -----------------------------------------------------------------------------------
        theta, P, Q = cp.Variable((N, 1)), cp.Variable((N, N)), cp.Variable((N, N))
        v = cp.Variable((N, 1), bounds=[v_min, v_max])
        p = cp.Variable(N, bounds=[p_min, p_max])
        q = cp.Variable(N, bounds=[q_min, q_max])
        C, S = cp.cos(theta - theta.T), cp.sin(theta - theta.T) 

        constr = [theta[0] == 0,  p == cp.sum(P, axis=1), q == cp.sum(Q, axis=1),
                P == cp.multiply(v @ v.T, cp.multiply(G, C) + cp.multiply(B, S)),
                Q == cp.multiply(v @ v.T, cp.multiply(G, S) - cp.multiply(B, C))]
        cost = (0.11 * p[0]**2 + 5 * p[0] + 150 + 0.085 * p[1]**2 + 1.2 * p[1] + 600 +
                0.1225 * p[2]**2 + p[2] + 335)
        prob = cp.Problem(cp.Minimize(cost), constr)

        # -----------------------------------------------------------------------------------
        #                            Solve prob 
        # -----------------------------------------------------------------------------------
        prob.solve(nlp=True, solver=cp.IPOPT, verbose=False)
                
        assert prob.status == cp.OPTIMAL
        assert np.abs(prob.value - 3087.84) / prob.value <= 1e-4

        checker = DerivativeChecker(prob)
        checker.run_and_assert()