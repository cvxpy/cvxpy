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
from cvxpy.reductions.solvers.bisection import bisect
import numpy as np

x = cp.Variable()
expr = cp.ceil(x)
assert expr.is_dqcp()
assert expr.is_quasiconvex()
assert expr.is_quasiconcave()
assert not expr.is_dcp()

y = cp.Variable()
constr = [cp.hstack([expr, y]) <= cp.hstack([20, 20])]
prob = cp.Problem(cp.Minimize(expr), constr + [x >= 12, x <= 17])
print(prob)
assert prob.is_dqcp()
assert not prob.is_dcp()
assert not prob.is_dgp()

red = cp.Dqcp2Dcp(prob)
reduced = red.reduce()

print(reduced)
print(reduced.parameters())

t = reduced.parameters()[0]
soln = bisect(red._bisection_data, verbose=True)
print(soln)
prob.unpack(soln)
print(prob.value)
print(x.value)

print("Solving QCP through problem.solve ...")
prob = cp.Problem(cp.Minimize(expr), [x >= 12, x <= 17])
prob.solve(qcp=True, verbose=True)
print(prob.value)
print(x.value)
