"""
Copyright 2022, the CVXPY authors

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
# Taken from CVX website http://cvxr.com/cvx/examples/
# An example for our contributing atoms tr_inv
# Created by Jianping Cai

# Solves the following SDP problem:
#           minimize    trace(inv(X))
#               s.t.    X is PSD
#                       trace(X)==1

import cvxpy as cp

T = 5

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.0
X = cp.Variable((T, T), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    cp.trace(X) == 1
]

prob = cp.Problem(cp.Minimize(cp.tr_inv(X)), constraints)
prob.solve(verbose=True)

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)
