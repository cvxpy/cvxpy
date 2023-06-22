from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class StandardTestLPs(BaseTest):

    def test_lp_1(self, solver='ECOS'):
        # typical LP
        sth = STH.lp_1()
        sth.solve(solver)
        sth.check_stationary_lagrangian(places=4)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_2(self, solver='ECOS'):
        # typical LP
        sth = STH.lp_2()
        sth.solve(solver)
        sth.check_stationary_lagrangian(places=2)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_3(self, solver='ECOS'):
        # unbounded LP
        sth = STH.lp_3()
        sth.solve(solver)
        sth.check_stationary_lagrangian(4, expect=False)
        sth.verify_objective(places=4)

    def test_lp_4(self, solver='ECOS'):
        # infeasible LP
        sth = STH.lp_4()
        sth.solve(solver)
        sth.check_stationary_lagrangian(4, expect=False)
        sth.verify_objective(places=4)

    def test_lp_5(self, solver='ECOS'):
        # LP with redundant constraints
        sth = STH.lp_5()
        sth.solve(solver)
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)
        sth.check_stationary_lagrangian(places=2)
