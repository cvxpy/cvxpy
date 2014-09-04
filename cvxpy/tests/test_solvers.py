from cvxpy import *
from base_test import BaseTest

class TestSolvers(BaseTest):
    """ Unit tests for solver specific behavior. """
    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    def test_solver_errors(self):
        """Tests that solver errors throw an exception.
        """
        # For some reason CVXOPT can't handle this problem.
        expr = 500*self.a + square(self.a)
        prob = Problem(Minimize(expr))

        with self.assertRaises(Exception) as cm:
            prob.solve(solver=CVXOPT)
        self.assertEqual(str(cm.exception),
            "Solver 'CVXOPT' failed. Try another solver.")

    def test_ecos_options(self):
        """Test that all the ECOS solver options work.
        """
        # Test ecos
        # feastol, abstol, reltol, feastol_inacc, abstol_inacc, and reltol_inacc for tolerance values
        # max_iters for the maximum number of iterations,
        EPS = 1e-4
        prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
        prob.solve(solver=ECOS, feastol=EPS, abstol=EPS, reltol=EPS,
                   feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                   max_iters=20, verbose=True)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_scs_options(self):
        """Test that all the SCS solver options work.
        """
        # Test SCS
        # MAX_ITERS, EPS, ALPHA, UNDET_TOL, VERBOSE, and NORMALIZE.
        # If opts is missing, then the algorithm uses default settings.
        # USE_INDIRECT = True
        EPS = 1e-4
        prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
        prob.solve(solver=SCS, max_iters=50, eps=EPS, alpha=EPS,
                   verbose=True, normalize=True, use_indirect=False)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_cvxopt_options(self):
        """Test that all the CVXOPT solver options work.
        """
        # TODO race condition when changing these values.
        # 'maxiters'
        # maximum number of iterations (default: 100).
        # 'abstol'
        # absolute accuracy (default: 1e-7).
        # 'reltol'
        # relative accuracy (default: 1e-6).
        # 'feastol'
        # tolerance for feasibility conditions (default: 1e-7).
        # 'refinement'
        # number of iterative refinement steps when solving KKT equations (default: 0 if the problem has no second-order cone or matrix inequality constraints; 1 otherwise).
        EPS = 1e-7
        prob = Problem(Minimize(norm(self.x, 1)), [self.x == 0])
        prob.solve(solver=CVXOPT, feastol=EPS, abstol=EPS, reltol=EPS,
                   max_iters=20, verbose=True)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])
