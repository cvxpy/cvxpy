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
from multiprocessing import Pool

# Problem data.
m = 100
n = 75
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
gamma = 0.1

NUM_PROCS = 4

def prox(args):
    f, v = args
    f += (rho/2)*sum_squares(x - v)
    Problem(Minimize(f)).solve()
    return x.value

# Setup problem.
rho = 1.0
x = Variable(n)
funcs = [sum_squares(A*x - b), gamma * norm(x, 1)]
ui = [np.zeros((n, 1)) for func in funcs]
xbar = np.zeros((n, 1))
pool = Pool(NUM_PROCS)
# ADMM loop.
for i in range(50):
    prox_args = [xbar - u for u in ui]
    xi = pool.map(prox, zip(funcs, prox_args))
    xbar = sum(xi)/len(xi)
    ui = [u + x_ - xbar for x_, u in zip(xi, ui)]

# Compare ADMM with standard solver.
prob = Problem(Minimize(sum(funcs)))
result = prob.solve()
print("ADMM best", (sum_squares(np.dot(A, xbar) - b) + gamma * norm(xbar, 1)).value)
print("ECOS best", result)
