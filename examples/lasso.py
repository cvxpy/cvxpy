"""
Copyright 2013 Steven Diamond

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

from cvxpy import *
import numpy as np
import cvxopt
from multiprocessing import Pool

# Problem data.
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(n)
gamma = Parameter(nonneg=True)

# Construct the problem.
x = Variable(m)
objective = Minimize(sum_squares(A*x - b) + gamma*norm(x, 1))
p = Problem(objective)

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

gammas = np.logspace(-1, 2, num=100)
# Serial computation.
x_values = [get_x(value) for value in gammas]

# Parallel computation.
pool = Pool(processes = 4)
par_x = pool.map(get_x, gammas)

for v1,v2 in zip(x_values, par_x):
    if np.linalg.norm(v1 - v2) > 1e-5:
        print("error")
