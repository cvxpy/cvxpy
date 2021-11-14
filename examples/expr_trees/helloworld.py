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

#!/usr/bin/env python

import random
from math import exp, pi, sqrt

import numpy as np

from cvxpy import SCS, Minimize, Problem, Variable, conv, norm


def gauss(n: float = 11,sigma: float = 1):
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

print(norm(signal - x.value, 1).value)
