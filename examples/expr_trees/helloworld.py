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
