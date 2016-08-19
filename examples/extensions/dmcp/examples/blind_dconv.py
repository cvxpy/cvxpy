__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

n = 100
m = 40
y = Variable(m)
x = Variable(n)

x0 = np.zeros((n,1))
x0[n/2+3:n/2+4] = 1
x0[n/2+23:n/2+24] = 0.8
x0[n/2+43:n/2+44] = 0.6

t = np.linspace(-2,2,m)
y0 = np.exp(-np.square(t)*2)
d = np.convolve(np.array(x0).flatten(),np.array(y0).flatten())

cost = norm(conv(y,x)-d) + 0.15*norm(x,1)
prob = Problem(Minimize(cost), [normInf(y) <= 1])

x.value = np.ones((n,1))
y.value = np.ones((m,1))
prob.solve(method = 'bcd')

plt.plot(np.array(abs(x).value).flatten(),'b-o')
plt.plot(np.array(abs(y).value).flatten(),'c-s')
plt.plot(d,'g-')
plt.plot(x0,'r--', linewidth = 2)
plt.plot(y0,'m-.', linewidth = 2)
print norm(conv(x,y)-d).value
plt.legend(["$x$", "$y$","$d$", "ground truth $x_0$","ground truth $y_0$"])
plt.show()
