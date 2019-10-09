"""
Copyright 2019 Shane Barratt

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

n = 10
P = np.random.randn(n, n)
P = P @ P.T + 1e-3*np.eye(n)
x = cp.Variable(n,)
t = cp.Variable(1,)

x.value = np.ones(n)
t.value = np.ones(1)*1

quad_over_lin = lambda x, t: cp.perspective(x, t, atom=lambda x: cp.quad_form(x, P))
objective = quad_over_lin(x, t)

prob = cp.Problem(cp.Minimize(objective), [x >= 1.0, t <= 1.0])
result = prob.solve(solver=cp.SCS)

print (result)
print (objective.value)
print (x.value)
print (t.value)


prob = cp.Problem(cp.Minimize(cp.quad_over_lin(np.linalg.cholesky(P).T * x, t)), [x >= 1.0, t <= 1.0])
result = prob.solve(solver=cp.SCS)

print (result)
print (x.value)