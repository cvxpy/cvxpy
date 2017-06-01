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

#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random

from math import pi, sqrt, exp

def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

np.random.seed(5)
random.seed(5)
DENSITY = 0.1
n = 1000
x = Variable(n)
# Create sparse signal.
signal = np.zeros(n)
for i in range(n):
    if random.random() < DENSITY:
        signal[i] = random.uniform(1, 100)

# Gaussian kernel.
m = 100
kernel = gauss(m)

# Noisy signal.
noisy_signal = conv(kernel, signal).value + np.random.normal(n+m-1)

obj = norm(conv(kernel, x) - noisy_signal)
constraints = [x >= 0]
prob = Problem(Minimize(obj), constraints)
result = prob.solve(solver=SCS, verbose=True)

print norm(signal - x.value, 1).value
