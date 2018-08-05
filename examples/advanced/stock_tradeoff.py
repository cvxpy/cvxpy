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

import cvxopt
import numpy
from cvxpy import *
from multiprocessing import Pool
from pylab import figure, show
import math

num_assets = 100
num_factors = 20

mu = cvxopt.exp( cvxopt.normal(num_assets) )
F = cvxopt.normal(num_assets, num_factors)
D = cvxopt.spdiag( cvxopt.uniform(num_assets) )
x = Variable(num_assets)
gamma = Parameter(nonneg=True)

expected_return = mu.T * x
variance = square(norm2(F.T*x)) + square(norm2(D*x))

# construct portfolio optimization problem *once*
p = Problem(
    Maximize(expected_return - gamma * variance),
    [sum(x) == 1, x >= 0]
)

# encapsulate the allocation function
def allocate(gamma_value):
    gamma.value = gamma_value
    p.solve()
    w = x.value
    expected_return, risk = mu.T*w, w.T*(F*F.T + D*D)*w
    return (expected_return[0], math.sqrt(risk[0]))

# create a pool of workers and a grid of gamma values
pool = Pool(processes = 4)
gammas = numpy.logspace(-1, 2, num=100)

# compute allocation in parallel
mu, sqrt_sigma = zip(*pool.map(allocate, gammas))

# plot the result
fig = figure(1)
ax = fig.add_subplot(111)
ax.plot(sqrt_sigma, mu)
ax.set_ylabel('expected return')
ax.set_xlabel('portfolio risk')

show()
