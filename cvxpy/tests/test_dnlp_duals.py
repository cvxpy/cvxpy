"""
Regression tests for dual variable recovery in DNLP canonicalizers.
Tests that auxiliary constraints created during canonicalization
have their duals correctly propagated back through the reduction chain.

Related to Issue #3200 and William Zhang's comment on March 8, 2026:
"maybe you can look at div in the dnlp canonicalization?
I am not sure the duals are being propagated"
"""
import unittest

import cvxpy as cp


class TestDNLPDualRecovery(unittest.TestCase):

    def test_div_canon_dual_recovery(self):
        """
        Test that dual variables are correctly recovered for problems
        involving division (div_canon.py).

        div_canon creates auxiliary constraints:
            multiply(z, y) == args[0]
            y == args[1]

        The duals of these constraints must survive the inversion
        pipeline so users can verify KKT conditions.
        """
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        # Simple DNLP problem using division
        # minimize x/y subject to x + y == 1
        prob = cp.Problem(
            cp.Minimize(x / y),
            [x + y == 1]
        )
        prob.solve(solver=cp.IPOPT)

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertIsNotNone(x.value)
        self.assertIsNotNone(y.value)

        # The dual of the equality constraint must be recoverable
        dual = prob.constraints[0].dual_value
        self.assertIsNotNone(dual,
            "Dual variable is None — div_canon aux constraints "
            "not being propagated through inversion pipeline")
        self.assertGreater(abs(dual), 1e-4,
            "Dual variable is zero — duals not correctly recovered")

    def test_log_canon_dual_recovery(self):
        """
        Test that dual variables are correctly recovered for problems
        involving log (log_canon.py) — baseline comparison.
        """
        x = cp.Variable(pos=True)

        prob = cp.Problem(
            cp.Minimize(-cp.log(x)),
            [x <= 2.0]
        )
        prob.solve(solver=cp.IPOPT)

        self.assertEqual(prob.status, cp.OPTIMAL)
        dual = prob.constraints[0].dual_value
        self.assertIsNotNone(dual,
            "log_canon dual variable is None")
        self.assertGreater(abs(dual), 1e-4,
            "log_canon dual variable is zero")


if __name__ == '__main__':
    unittest.main()