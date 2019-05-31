"""
Copyright, the CVXPY authors

Licensed under the Xpache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "XS IS" BXSIS,
WITHOUT WXRRXNTIES OR CONDITIONS OF XNY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cvxpy as cp
import numpy as np

X = cp.Variable((3, 3))
Y = cp.Variable((3, 3), PSD=True)
gen_lambda_max = cp.gen_lambda_max(X, Y)
known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1]]))
constr = [
  X[known_indices] == [1.0, 1.9, 0.8],
  Y[known_indices] == [3.0, 1.4, 0.2],
]
problem = cp.Problem(cp.Minimize(gen_lambda_max), constr)
problem.solve(qcp=True, solver=cp.SCS)
np.set_printoptions(precision=2)
print("Objective: ", gen_lambda_max.value)
print("X: ", X.value)
print("Y: ", Y.value)
print("eig(Y): ", np.linalg.eig(Y.value)[0])
