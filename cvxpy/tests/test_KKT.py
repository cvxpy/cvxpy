import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class StandardTestLPs(BaseTest):

    def test_lp_1(self):
        # typical LP
        sth = STH.lp_1()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver = 'MOSEK')
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
            sth.solve(solver = 'MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(places=2)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_3(self):
        # unbounded LP
        sth = STH.lp_3()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver = 'MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(4, expect=False)
        sth.verify_objective(places=4)

    def test_lp_4(self):
        # infeasible LP
        sth = STH.lp_4()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver = 'MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(4, expect=False)
        sth.verify_objective(places=4)

    def test_lp_5(self):
        # LP with redundant constraints
        sth = STH.lp_5()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver = 'MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)
        sth.check_stationary_lagrangian(places=2)

    def test_lp_6(self):
        # unbounded problem --- cannot check stationarity
        sth = STH.lp_6()
        if 'MOSEK' in cp.installed_solvers():
            sth.solve(solver = 'MOSEK')
        else:
            sth.solve(solver='ECOS')
        sth.check_stationary_lagrangian(4, expect = False)
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
