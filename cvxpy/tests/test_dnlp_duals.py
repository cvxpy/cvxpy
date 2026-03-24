import unittest
import pytest
import cvxpy as cp


@pytest.mark.skipif(
    cp.IPOPT not in cp.installed_solvers(),
    reason="IPOPT not installed"
)
class TestDNLPDualRecovery(unittest.TestCase):

    def test_div_canon_dual_recovery(self):
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x / y), [x + y == 1])
        prob.solve(solver=cp.IPOPT, nlp=True)
        self.assertEqual(prob.status, cp.OPTIMAL)
        dual = prob.constraints[0].dual_value
        self.assertIsNotNone(dual)
        self.assertGreater(abs(dual), 1e-4)

    def test_log_canon_dual_recovery(self):
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(-cp.log(x)), [x <= 2.0])
        prob.solve(solver=cp.IPOPT, nlp=True)
        self.assertEqual(prob.status, cp.OPTIMAL)
        dual = prob.constraints[0].dual_value
        self.assertIsNotNone(dual)
        self.assertGreater(abs(dual), 1e-4)


if __name__ == "__main__":
    unittest.main()
    
