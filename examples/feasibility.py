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

from cvxpy import Minimize, Problem, Variable, INFEASIBLE, sum

# Example: Verify the feasibility of constraints, a naive and an improved version

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

import numpy as np
if __name__ == "__main__":
	# Create variables and constraints
	x = Variable(5)
	A = np.random.randn(3, 5)
	b = np.random.randn(3)

	constraints = [A @ x >= b]

	# Both function will always output the same value
	assert is_feasible_naive(x, constraints) == is_feasible_improved(constraints)
