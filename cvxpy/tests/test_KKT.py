import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class StandardTestLPs(BaseTest):

    def test_lp_1(self):
        # typical LP
        sth = STH.lp_1()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(places=4)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_2(self):
        # typical LP
        sth = STH.lp_2()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(places=2)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_5(self):
        # LP with redundant constraints
        sth = STH.lp_5()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)
        sth.check_stationary_lagrangian(places=2)


class StandardTestQPs(BaseTest):

    def test_qp_0(self):
        sth = STH.qp_0()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
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
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_1(self):
        sth = STH.socp_1()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_2(self):
        sth = STH.socp_2()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth

    def test_socp_3ax0(self):
        sth = STH.socp_3(axis=0)
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth


    def test_socp_3ax1(self):
        sth = STH.socp_3(axis=1)
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.check_complementarity(places=4)
        sth.verify_dual_values(places=4)
        sth.check_stationary_lagrangian(places=4)
        return sth


class StandardTestPCPs(BaseTest):

    def test_pcp_1(self, places: int = 3):
        sth = STH.pcp_1()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_2(self, places: int = 3):
        sth = STH.pcp_2()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_3(self, places: int = 3):
        sth = STH.pcp_3()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver='MOSEK')
        else:
            sth.solve(solver='SCS')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        sth.verify_primal_values(places)
        sth.check_complementarity(places)
        sth.verify_dual_values(places)
        sth.check_stationary_lagrangian(places)
        return sth
