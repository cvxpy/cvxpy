import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.kron import kron

from cvxpy.atoms.affine.partial_trace import partial_trace
from cvxpy.tests import solver_test_helpers as STH

def applychan(chan: np.array, rho: cp.Variable, rep: str, dim: tuple[int, int]):
    tol = 1e-10

    dimA, dimB, dimE = None, None, None
    match rep:
        case 'choi2':
            dimA, dimB = dim
        case 'isom':
            dimA = chan.shape[1]
            dimB = dim[1]
            dimE = int(chan.shape[0]/dimB)
            pass

    match rep:
        case 'choi2':
            arg = chan @ kron(rho.T, np.eye(dimB))
            rho_out = partial_trace(arg, [dimA, dimB], 0)
            return rho_out
        case 'isom':
            rho_out = partial_trace(chan @ rho @ chan.conj().T, [dimB, dimE], 1)
            return rho_out

def randH(n: int):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.conj().T)/2

def randRho(n: int):
    p = 10 * randH(n)
    p = (p @ p.conj().T)/np.trace(p @ p.conj().T)
    return p

class Test_quantum_rel_entr:
    """
    Test class for `quantum_rel_entr` & `quantum_cond_entr`
    - All of the reference solutions for the problem come from equivalent
    CVXQUAD implementations
    - These problems also show up in a Marimo notebook linked on the CVXPY docs
    """
    if 'MOSEK' in cp.installed_solvers():
        SOLVE_ARGS = {'solver': 'MOSEK', 'verbose': True}
    else:
        SOLVE_ARGS = {'solver': 'SCS', 'eps': 1e-6, 'max_iters': 500_000,
                      'verbose': True}

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
        rho = np.array([
        [0.0695342 - 6.70188253e-20j, 0.05068878 - 3.93324547e-02j, -0.02266915 - 5.05888301e-03j, -0.07177916 - 9.58371965e-03j],
        [0.05068878 + 3.93324547e-02j, 0.12891079 - 3.18040357e-18j, 0.00853936 + 2.22083687e-02j, -0.12811657 - 1.45536424e-01j],
        [-0.02266915 + 5.05888301e-03j, 0.00853936 - 2.22083687e-02j, 0.46613619 + 3.02700898e-18j, -0.05220245 - 2.68353102e-03j],
        [-0.07177916 + 9.58371965e-03j, -0.12811657 + 1.45536424e-01j, -0.05220245 + 2.68353102e-03j, 0.33541883 + 2.20413415e-19j]])

        tau = cp.Variable(shape=(na * nb, na * nb), hermitian=True)
        expect_tau = np.array([[0.0692 + 0.0000j, 0.0383 - 0.0310j, -0.0304 - 0.0143j, -0.0507 - 0.0019j],
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

        obj = cp.Maximize((cp.von_neumann_entr(p1 * rho1 + p2 * rho2) - p1 * H1 - p2 * H2)/np.log(2))
        obj_expect = 0.60088
        obj_pair = (obj, obj_expect)

        cons1 = p1 >= 0
        cons2 = p2 >= 0
        cons3 = p1 + p2 == 1
        cons_pair = [(cons1, None), (cons2, None), (cons3, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, cons_pair)

        return sth

    @staticmethod
    def make_test_4():
        """
        % Quantum capacity of degradable channels

        % Example: amplitude damping channel
        % na = channel input dimension
        % nb = channel output dimension
        % ne = channel environment dimension
        % nf = degrading map environment dimension
        """
        na, nb, ne, nf = (2, 2, 2, 2)
        AD = lambda gamma: np.array([[1, 0],[0, np.sqrt(gamma)],[0, np.sqrt(1-gamma)],[0, 0]])
        gamma = 0.2
        U = AD(gamma)

        W = AD((1-2*gamma)/(1-gamma))

        Ic = lambda rho: cp.quantum_cond_entr(
            W @ applychan(U, rho, 'isom', (na, nb)) @ W.conj().T,
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

    @staticmethod
    def make_test_5():
        """
        % Entanglement-assisted classical capacity of a quantum channel

        % Dimensions of input, output, and environment spaces of channel
        """
        na, nb, ne = (2, 2, 2)
        AD = lambda gamma: np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])
        U = AD(0.2)

        rho = cp.Variable(shape=(na, na), hermitian=True)
        rho_expect = np.array([[0.5185, 0],
                               [0, 0.4815]])
        var_pairs = [(rho, rho_expect)]

        obj = cp.Maximize((cp.quantum_cond_entr(U @ rho @ U.conj().T, [nb, ne]) +
                        cp.von_neumann_entr(cp.partial_trace(U @ rho @ U.conj().T, [nb, ne], 1)))/np.log(2))
        obj_expect = -np.inf
        obj_pair = (obj, obj_expect)

        cons1 = rho >> 0
        cons2 = cp.trace(rho) == 1
        cons_pairs = [(cons1, None), (cons2, None)]

        sth = STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)

        return sth

    def test_1(self):
        sth = Test_quantum_rel_entr.make_test_1()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_2(self):
        sth = Test_quantum_rel_entr.make_test_2()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_3(self):
        sth = Test_quantum_rel_entr.make_test_3()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_4(self):
        sth = Test_quantum_rel_entr.make_test_4()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)

    def test_5(self):
        sth = Test_quantum_rel_entr.make_test_5()
        sth.solve(**self.SOLVE_ARGS)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
