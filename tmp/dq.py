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

x = cp.Variable()
y = cp.Variable(pos=True)
objective_fn = -cp.sqrt(x)/y
objective = cp.Minimize(objective_fn)
constraints = [cp.exp(x) <= y]
problem = cp.Problem(objective, constraints)
problem.solve(qcp=True)
print("Optimal value: ", problem.value)
print("x: ", x.value)
print("y: ", y.value)
