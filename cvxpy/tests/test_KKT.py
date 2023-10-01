import numpy as np

import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class TestKKT_LPs(BaseTest):

    def test_lp_1(self, places=4):
        # typical LP
        sth = STH.lp_1()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)

    def test_lp_2(self, places=4):
        # typical LP
        sth = STH.lp_2()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)

    def test_lp_5(self, places=4):
        # LP with redundant constraints
        sth = STH.lp_5()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)


class TestKKT_QPs(BaseTest):

    def test_qp_0(self, places=4):
        sth = STH.qp_0()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth


class TestKKT_SOCPs(BaseTest):

    def test_socp_0(self, places=4):
        sth = STH.socp_0()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_socp_1(self, places=4):
        sth = STH.socp_1()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_socp_2(self, places=4):
        sth = STH.socp_2()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_socp_3ax0(self, places=4):
        sth = STH.socp_3(axis=0)
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth


    def test_socp_3ax1(self, places=4):
        sth = STH.socp_3(axis=1)
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth


class TestKKT_ECPs(BaseTest):

    def test_expcone_1(self, places = 4):
        sth = STH.expcone_1()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

class TestKKT_SDPs(BaseTest):

    def test_sdp_1min(self, places=4):
        sth = STH.sdp_1('min')
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_sdp_1max(self, places=4):
        sth = STH.sdp_1('max')
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_sdp_2(self, places=4):
        sth = STH.sdp_2()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth


class TestKKT_PCPs(BaseTest):

    @staticmethod
    def non_vec_pow_nd() -> STH.SolverTestHelper:
        # A contrived `PowConeND` projection problem, tests the dual value
        # implementation for non-vectorized `PowConeND` constraints
        dims = 6
        x = cp.Variable(shape=(dims,))
        x_power = cp.Variable(shape=(dims,))
        alpha = np.array([0.19255292, 0.39811507, 0.17199319, 0.02634437, 0.21099446])
        pow_con = cp.PowConeND(x_power[:dims - 1], x_power[dims - 1], alpha)
        obj = cp.Minimize(cp.norm(x - x_power))
        cons = [pow_con,
                cp.bmat([[87 * x[0], x[1], x[2] / 3],
                          [100.0, 4 * 1e2, x[3] * 78],
                          [23 * x[4], 1e3, x[5]/144]]) >> 0,
                cp.ExpCone(x[0], x[3], x[5])]
        obj_pair = (obj, 67030.289)
        con_pairs = [(con, None) for con in cons]
        var_pairs = [(x, np.array([1.88001162e-02, -4.88454651e+01, 8.57595590e+05, 1.02810213e-03,
                                    -1.24261412e+04, 9.00144344e+04])),
                     (x_power, np.array([19633.20414201, 28205.58222152,
                                         857996.79202862, 7261.49580318,
                                         15256.79776484, 40114.03651347]))]
        return STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)

    @staticmethod
    def vec_pow_nd() -> STH.SolverTestHelper:
        # A contrived `PowConeND` projection problem, tests the dual value
        # implementation for vectorized `PowConeND` constraints
        axis = 1
        dims = 6
        x = cp.Variable(shape=(dims,))
        x_power1 = cp.Variable(shape=(dims,))
        x_power2 = cp.Variable(shape=(dims,))
        x_power3 = cp.Variable(shape=(dims,))
        alpha1 = np.array([0.16258832, 0.00095657, 0.39366008, 0.13756387, 0.30523115])
        alpha2 = np.array([0.00652187, 0.2102224 , 0.46568224, 0.18612647, 0.13144702])
        alpha3 = np.array([0.0707463 , 0.15574321, 0.03874102, 0.51298334, 0.22178612])
        alpha = np.vstack([alpha1, alpha2, alpha3])
        W = cp.vstack([x_power1[:dims - 1], x_power2[:dims - 1], x_power3[:dims - 1]])
        z = cp.hstack([x_power1[dims - 1], x_power2[dims - 1], x_power3[dims-1]])
        pow_con = cp.PowConeND(W, z, alpha, axis=axis)
        obj = cp.Minimize(cp.norm(x - 10 * x_power1 - 2 * x_power2 - 15 * x_power3))
        cons = [pow_con,
                cp.bmat([[87 * x[0], x[1], x[2] / 3],
                          [100.0, 4 * 1e2, x[3] * 78],
                          [23 * x[4], 1e3, x[5]/144]]) >> 0,
                cp.ExpCone(x[0], x[3], x[5])]
        obj_pair = (obj, None)
        con_pairs = [(con, None) for con in cons]
        var_pairs = [(x, np.array([1.08377501e-01, 2.28254565e+01,
                                   1.06350094e+06, 6.59645560e-03,
                                   -1.54063784e+04, 9.00926375e+04])),
                     (x_power1, np.array([1.07660729e-03, 2.62701672e-05,
                                          1.62230262e-02, 2.12816300e-04,
                                          2.19779775e-04, 1.25672222e-03])),
                     (x_power2, np.array([1032.37388779, 5868.71051659,
                                          531893.56084737, 5517.13267126,
                                          2176.03757127, 41063.97799287])),
                     (x_power3, np.array([1.22635012e-04, 3.44641582e-05,
                                          9.74881635e-04, 1.16748278e-04,
                                          3.40069711e-05, 1.16813104e-05]))]
        return STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)

    def test_pcp_1(self, places: int = 4):
        sth = STH.pcp_1()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_2(self, places: int = 4):
        sth = STH.pcp_2()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_3(self, places: int = 4):
        sth = STH.pcp_3()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    # Tests for verifying the dual value implementation of `PowConeND`
    def test_pcp_4(self, places: int=3):
        sth = self.non_vec_pow_nd()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_5(self, places: int=3):
        sth = self.vec_pow_nd()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth


class TestKKT_Flags(BaseTest):
    """
    A class with tests for testing out the incorporation of
    implicit constraints within `check_stationarity_lagrangian`
    """

    @staticmethod
    def nsd_flag() -> STH.SolverTestHelper:
        """
        Tests NSD flag
        Reference values via MOSEK
        Version: 10.0.46
        """
        X = cp.Variable(shape=(3,3), NSD=True)
        obj = cp.Maximize(cp.lambda_min(X))
        cons = [X[0, 1] == 123]
        con_pairs = [
            (cons[0], None),
        ]
        var_pairs = [(X, np.array([[-123.,  123.,    0.],
                                   [ 123., -123.,    0.],
                                   [   0.,    0., -123.]]))]
        obj_pair = (obj, -246.0000000000658)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def psd_flag() -> STH.SolverTestHelper:
        """
        Tests PSD flag
        Reference values via MOSEK
        Version: 10.0.46
        """
        X = cp.Variable(shape=(4,4), PSD=True)
        obj = cp.Minimize(cp.log_sum_exp(X))
        cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
            (cons[2], None),
        ]
        var_pairs = [(X, np.array([[ 4.00000001,  4.        , -1.05467058, -1.05467058],
                                   [ 4.        ,  4.00000001, -1.05467058, -1.05467058],
                                   [-1.05467058, -1.05467058,  0.27941584,  0.27674984],
                                   [-1.05467058, -1.05467058,  0.27674984,  0.27941584]]))]
        obj_pair = (obj, 5.422574709567284)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def symmetric_flag() -> STH.SolverTestHelper:
        """
        Tests symmetric flag
        Reference values via MOSEK
        Version: 10.0.46
        """
        X = cp.Variable(shape=(4,4), symmetric=True)
        obj = cp.Minimize(cp.log_sum_exp(X))
        cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
            (cons[2], None),
        ]
        var_pairs = [(X, np.array([[-3.74578525,  4.        , -3.30586268, -3.30586268],
                                   [ 4.        , -3.74578525, -3.30586268, -3.30586268],
                                   [-3.30586268, -3.30586268, -2.8684253 , -2.8684253 ],
                                   [-3.30586268, -3.30586268, -2.8684253 , -2.86842529]]))]
        obj_pair = (obj, 4.698332858812026)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def nonneg_flag() -> STH.SolverTestHelper:
        """
        Tests nonneg flag
        Reference values via MOSEK
        Version: 10.0.46
        """
        X = cp.Variable(shape=(4,4), nonneg=True)
        obj = cp.Minimize(cp.log_sum_exp(X))
        cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
            (cons[2], None),
        ]
        var_pairs = [(X, np.array([[1.19672119e-07, 4.00000000e+00, 1.19672119e-07, 1.19672119e-07],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309088e-08]]
                                  ))]
        obj_pair = (obj, 4.242738008082711)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def nonpos_flag() -> STH.SolverTestHelper:
        """
        Tests nonpos flag
        Reference values via MOSEK
        Version: 10.0.46
        """
        X = cp.Variable(shape=(3, 3), nonpos=True)
        obj = cp.Minimize(cp.norm2(X))
        cons = [cp.log_sum_exp(X) <= 2, cp.sum_smallest(X, 5) >= -10]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
        ]
        var_pairs = [(X, np.array([[-0.19722458, -0.19722458, -0.19722457],
                                   [-0.19722458, -0.19722458, -0.19722457],
                                   [-0.19722457, -0.19722457, -0.19722459]]))]
        obj_pair = (obj, 0.5916737242761841)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    """
    Only verifying the KKT conditions in these tests
    """
    def test_kkt_nsd_var(self, places=4):
        sth = TestKKT_Flags.nsd_flag()
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_kkt_psd_var(self, places=4):
        sth = TestKKT_Flags.psd_flag()
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_kkt_symmetric_var(self, places=4):
        sth = TestKKT_Flags.symmetric_flag()
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_kkt_nonneg_var(self, places=4):
        sth = TestKKT_Flags.nonneg_flag()
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_kkt_nonpos_var(self, places=4):
        sth = TestKKT_Flags.nonpos_flag()
        sth.solve(solver='SCS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth
