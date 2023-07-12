import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class StandardTestLPs(BaseTest):

    def test_lp_1(self):
        # typical LP
        sth = STH.lp_1()
        sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(places=4)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_2(self):
        # typical LP
        sth = STH.lp_2()
        sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(places=2)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_5(self):
        # LP with redundant constraints
        sth = STH.lp_5()
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)
        sth.check_stationary_lagrangian(places=2)


class StandardTestQPs(BaseTest):

    def test_qp_0(self):
        sth = STH.qp_0()
        sth.solve(solver='ECOS')
        sth.verify_primal_values(places=4)
        sth.verify_objective(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth


class StandardTestSOCPs(BaseTest):

    def test_socp_0(self):
        sth = STH.socp_0()
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_1(self):
        sth = STH.socp_1()
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_2(self):
        sth = STH.socp_2()
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_3ax0(self):
        sth = STH.socp_3(axis=0)
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth


    def test_socp_3ax1(self):
        sth = STH.socp_3(axis=1)
        sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth


class StandardTestSDPs(BaseTest):

    def test_sdp_1min(self):
        sth = STH.sdp_1('min')
        sth.solve(solver='SCS')
        sth.verify_objective(places=2)
        sth.check_primal_feasibility(places=2)
        sth.check_complementarity(places=2)
        sth.check_dual_domains(places=2)
        sth.check_stationary_lagrangian(places=2)
        return sth

    def test_sdp_1max(self):
        sth = STH.sdp_1('max')
        sth.solve(solver='SCS')
        sth.verify_objective(places=2)
        sth.check_primal_feasibility(places=2)
        sth.check_complementarity(places=2)
        sth.check_dual_domains(places=2)
        sth.check_stationary_lagrangian(places=2)
        return sth

    def test_sdp_2(self):
        # places is set to 3 rather than 4, because analytic solution isn't known.
        sth = STH.sdp_2()
        sth.solve(solver='SCS')
        sth.verify_objective(places=2)
        sth.check_primal_feasibility(places=2)
        sth.verify_primal_values(places=2)
        sth.check_complementarity(places=2)
        sth.check_dual_domains(places=2)
        sth.check_stationary_lagrangian(places=2)
        return sth


class StandardTestPCPs(BaseTest):

    def test_pcp_1(self, places: int = 2):
        sth = STH.pcp_1()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_2(self, places: int = 2):
        sth = STH.pcp_2()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_3(self, places: int = 2):
        sth = STH.pcp_3()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth


class StandardTestFlags(BaseTest):
    """
    A class with tests for testing out the incorporation of
    implicit constraints within `check_stationarity_lagrangian`
    """

    @staticmethod
    def tf_1() -> STH.SolverTestHelper:
        """
        Tests NSD flag
        """
        X = cp.Variable(shape=(3,3), NSD=True)
        obj = cp.Maximize(cp.lambda_min(X))
        cons = [X == X.T, X[0][1] == 123]
        con_pairs = [
            (cons[0], None),
            (cons[1], None)
        ]
        var_pairs = [(X, np.array([[-122.99997839,  122.99999997, 0],
                                   [ 122.99999997, -122.99997839, 0],
                                   [ 0, 0, -111.45713829]]))]
        obj_pair = (obj, -245.99997835783373)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def tf_2() -> STH.SolverTestHelper:
        """
        Tests PSD flag
        """
        X = cp.Variable(shape=(4,4), PSD=True)
        obj = cp.Minimize(cp.log_sum_exp(X))
        cons = [cp.norm2(X) <= 10, X[0][1] >= 4, X[0][1] <= 8]
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
    def tf_3() -> STH.SolverTestHelper:
        """
        Tests nonneg flag
        """
        X = cp.Variable(shape=(4,4), nonneg=True)
        obj = cp.Minimize(cp.log_sum_exp(X))
        cons = [cp.norm2(X) <= 10, X[0][1] >= 4, X[0][1] <= 8]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
            (cons[2], None),
        ]
        var_pairs = [(X, np.array([[1.19672119e-07, 4.00000000e+00, 1.19672119e-07, 1.19672119e-07],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08],
                                   [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309088e-08]]))]
        obj_pair = (obj, 4.242738008082711)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    @staticmethod
    def tf_4() -> STH.SolverTestHelper:
        """
        Tests nonpos flag
        """
        X = cp.Variable(shape=(3, 3), nonpos=True)
        obj = cp.Minimize(cp.norm2(X))
        cons = [cp.log_sum_exp(X) <= 2, cp.sum_smallest(X, 5) >= -10]
        con_pairs = [
            (cons[0], None),
            (cons[1], None),
        ]
        var_pairs = [(X, np.array([[-0.19722456, -0.19722456, -0.19722456],
                                   [-0.19722456, -0.19722456, -0.19722456],
                                   [-0.19722456, -0.19722456, -0.19722456]]))]
        obj_pair = (obj, 0.5916736882031846)
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_tf_1(self, places: int = 4):
        sth = StandardTestFlags.tf_1()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)

    def test_tf_2(self, places: int = 2):
        sth = StandardTestFlags.tf_2()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places=4)

    def test_tf_3(self, places: int = 4):
        sth = StandardTestFlags.tf_3()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)

    def test_tf_4(self, places: int = 4):
        sth = StandardTestFlags.tf_4()
        sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
