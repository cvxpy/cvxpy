"""
Copyright, the CVXPY authors

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

import time
from unittest import TestCase

import numpy as np

from cvxpy import INFEASIBLE, Minimize, Problem, Variable, sum


# Naive version with a simple dummy objective
def is_feasible_naive(vars, constraints):
	problem = Problem(Minimize(sum(vars)), constraints)

	problem.solve()

	return problem.status != INFEASIBLE

# Improved version: cvxpy allows for objectives that don't depend on the variables
# this will stop as soon as a first solution is encountered
def is_feasible_improved(constraints):
	problem = Problem(Minimize(0), constraints)

	problem.solve()

	return problem.status != INFEASIBLE

class TestFeasibilty(TestCase):
    """ Unit tests of the improved feasibility checking method (See feasibility example). """

    def test_feasibility_functions_equal(self) -> None:
        for _ in range(10):
            np.random.seed(27)
            first_dim = np.random.randint(1, 10)
            second_dim = np.random.randint(1, 10)
            x = Variable(first_dim)
            A = np.random.randn(second_dim, first_dim)
            b = np.random.randn(second_dim)

            constraints = [A @ x >= b]

            self.assertEqual(is_feasible_naive(x, constraints), is_feasible_improved(constraints))

    def test_improved_function_speedup(self) -> None:
        naive_times = []
        improved_times = []

        for _ in range(10):
            np.random.seed(27)
            first_dim = np.random.randint(1, 10)
            second_dim = np.random.randint(1, 10)
            x = Variable(first_dim)
            A = np.random.randn(second_dim, first_dim)
            b = np.random.randn(second_dim)

            constraints = [A @ x >= b]

            start = time.time()
            is_feasible_naive(x, constraints)
            end = time.time()

            naive_times.append(end-start)

            x = Variable(first_dim)
            constraints = [A @ x >= b]

            start = time.time()
            is_feasible_improved(constraints)
            end = time.time()

            improved_times.append(end-start)

        self.assertGreater(sum(naive_times), sum(improved_times))
