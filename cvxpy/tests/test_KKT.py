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
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth

    def test_socp_2(self, places=4):
        sth = STH.socp_2()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth

    def test_socp_3ax0(self, places=4):
        sth = STH.socp_3(axis=0)
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth


    def test_socp_3ax1(self, places=4):
        sth = STH.socp_3(axis=1)
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth


class TestKKT_ECPs(BaseTest):

    def test_expcone_1(self, places = 4):
        sth = STH.expcone_1()
        sth.solve(solver='ECOS')
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
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

    def test_pcp_1(self, places: int = 4):
        sth = STH.pcp_1()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_2(self, places: int = 4):
        sth = STH.pcp_2()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_3(self, places: int = 4):
        sth = STH.pcp_3()
        sth.solve(solver='SCS', eps=1e-6)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        try:
            sth.check_dual_domains(places)
        except NotImplementedError:
            pass
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
