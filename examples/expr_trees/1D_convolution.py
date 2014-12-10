#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random
from math import pi, sqrt, exp

def gauss(n=11,sigma=1, scale=1):
    r = range(-int(n/2),int(n/2)+1)
    return [scale / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

np.random.seed(5)
random.seed(5)
n = 1000
DENSITY = 6.0/n
x = Variable(n)
# Create sparse signal.
HEIGHT = 100
true_x = np.zeros((n,1))
nnz = 0
for i in range(n):
    if random.random() < DENSITY:
        true_x[i] = random.uniform(0, HEIGHT)
        nnz += 1

# Gaussian kernel.
m = 101
kernel = gauss(m, m/10, 1)

# Noisy signal.
SNR = 10
signal = conv(kernel, true_x)
sigma = norm(signal,2).value/(SNR*sqrt(n+m-1))
noise = np.random.normal(scale=sigma, size=n+m-1)
print("SNR", norm(signal,2).value/norm(noise,2).value)
noisy_signal = signal + noise

gamma = Parameter(sign="positive")
fit = norm(conv(kernel, x) - noisy_signal, 2)
constraints = [x >= 0]
prob = Problem(Minimize(fit),
               constraints)
result = prob.solve(solver=ECOS, verbose=True)
# result = prob.solve(solver=SCS,
#                     verbose=True,
#                     max_iters=2500,
#                     eps=1e-3,
#                     use_indirect=True)
# print("true signal fit", fit.value)
# result = prob.solve(solver=SCS_MAT_FREE,
#                     verbose=True,
#                     max_iters=500,
#                     equil_steps=1,
#                     eps=1e-3)
print("recovered x fit", fit.value)


from numpy.fft import fft, ifft
print(np.array(kernel).shape)
k2 = np.hstack([np.array(kernel), np.zeros(n-1)])
x2 = np.hstack([true_x[:,0], np.zeros(m-1)])
elems = np.square(np.absolute(fft(k2)*fft(x2)))
print(fft(k2)*fft(x2) - fft(signal.value)[:,0])
print((ifft(fft(k2)*fft(x2)) - signal).value[:,0])
test = np.sqrt(np.sum(elems))
print test/(n+m-1)
print test/(n+m-1) - norm(signal,2).value

# Plot result and fit.
# Assumes convolution kernel is centered around m/2.
t = range(n+m-1)
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(2, 1, 1)
true_x_padded = np.vstack([np.zeros((m/2,1)), true_x, np.zeros((m/2,1))])
x_padded = np.vstack([np.zeros((m/2,1)), x.value[:,0], np.zeros((m/2,1))])
plt.plot(t, true_x_padded, label="true x")
plt.plot(t, x_padded, label="recovered x")
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.plot(t, np.asarray(noisy_signal.value[:, 0]), label="signal")
plt.legend(loc='upper right')
plt.show()
