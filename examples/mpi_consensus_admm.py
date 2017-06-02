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

# Consensus ADMM
# See ADMM paper section 7

# xi = argmin [f(xi) + (rho/2)norm(xi - xbar + ui)]
# ui = ui + xi - xbar

from cvxpy import *
from multiprocessing import Pool
import operator as op
import numpy as np
from numpy.random import randn
import dill

# Initialize the problem.
n = 1000
m = 500
rho = 1.0
xbar = 0

# No! Distribute problem objects.
# Use MPI.
# Create function to perform local update from penalty f.
def create_update(f):
    x = Variable(n)
    u = Parameter(n)
    def local_update(xbar):
        # Update u.
        if x.value is None:
            u.value = np.zeros(n)
        else:
            u.value += x.value - xbar
        # Update x.
        obj = f(x) + (rho/2)*sum_squares(x - xbar + u)
        Problem(Minimize(obj)).solve()
        return x.value

    return local_update

# Penalty functions.
functions = map(dill.dumps,
    map(create_update, [
        lambda x: norm(randn(m, n)*x + randn(m), 2),
        lambda x: norm(randn(m, n)*x + randn(m), 2),
        lambda x: norm(randn(m, n)*x + randn(m), 2),
        lambda x: norm(randn(m, n)*x + randn(m), 2),
        lambda x: norm(x, 1),
    ])
)

# Do ADMM iterations in parallel.
def apply_f(args):
    f = dill.loads(args[0])
    return f(args[1])

pool = Pool(processes = len(functions))
for i in range(10):
    total = reduce(op.add,
        pool.map(apply_f, zip(functions, len(functions)*[xbar]))
    )
    xbar = total/len(functions)
    print i
