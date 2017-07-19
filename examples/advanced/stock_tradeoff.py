"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
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
