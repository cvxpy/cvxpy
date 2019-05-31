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

import cvxpy as cp
import numpy as np
np.set_printoptions(precision=2)

n = 10
np.random.seed(1)
A = np.random.randn(n, n)
x_star = np.random.randn(n)
b = A @ x_star
epsilon = 1e-2

x = cp.Variable(n)
objective_fn = cp.length(x)
mse = cp.sum_squares(A @ x - b)/n
problem = cp.Problem(cp.Minimize(objective_fn), [mse <= epsilon])
problem.solve(qcp=True)
print("Length of x: ", problem.value)
print("MSE: ", mse.value)
print("x: ", x.value)
print("x_star: ", x_star)
