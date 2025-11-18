"""
Copyright 2025 CVXPY developers  

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at  

    http://www.apache.org/licenses/LICENSE-2.0  

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.
"""

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestPowerAtom(BaseTest):
    """Unit tests for power atom."""

    def test_power_approx(self) -> None:
        """Test power atom with approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=True) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_no_approx(self) -> None:
        """Test power atom without approximation."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.SCS)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)

    def test_power_with_and_without_approx_low(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, -1.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value 

        constr = [
            cp.power(x, -1.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_mid(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 0.8, approx=True) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value 

        constr = [
            cp.power(x, 0.8, approx=False) >= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_high(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 4.5, approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value 

        constr = [
            cp.power(x, 4.5, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_with_and_without_approx_even(self) -> None:
        """Compare answers with and without approximation on the same problem."""
        x = cp.Variable(3)
        constr = [
            cp.power(x, 8,approx=True) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        x_approx = x.value 

        constr = [
            cp.power(x, 8, approx=False) <= np.ones(3),
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2] + (x[1] + x[2])**2), constr)
        prob.solve(solver=cp.CLARABEL)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertItemsAlmostEqual(x.value, x_approx, places=3)

    def test_power_no_approx_unsupported_solver(self) -> None:
        """Test power atom without approximation, using unsupported solver."""
        x = cp.Variable(3)
        constr = [cp.power(x, 3.3, approx=False) <= np.ones(3)]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] - x[2]), constr)
        prob.solve(solver=cp.CVXOPT)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -1.0, places=3)
        expected_x = np.array([0.0, 0.0, 1.0])
        self.assertItemsAlmostEqual(x.value, expected_x, places=3)


test = TestPowerAtom()
# test.test_power_with_and_without_approx_low()
test.test_power_no_approx_unsupported_solver()