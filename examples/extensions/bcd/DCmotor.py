__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

d = 2
m = 1
n = 150

u = Variable(m,n-1)
x = Variable(d,n)


u.value = np.ones((m,n-1))
x.value = np.zeros((d,n))

#u = NonNegative(1)
#u.value = 0.368

A0 = np.matrix([[-1,0],[0,-0.1]])*0.1
A1 = np.matrix([[0,-19],[0.1,0]])*0.1

#constr = [x[:,0]-1 == A0*np.ones((2,1))+A1*np.ones((2,1))*1, max_entries(abs(x[0,:])) <= 8]
constr = [x[:,0]-1 == 0, max_entries(abs(x[0,:])) <= 8]
for t in range(n-1):
    constr += [x[:,t+1]-x[:,t] == A0*x[:,t]+A1*x[:,t]*u[t]]
prob = Problem(Minimize(norm(x[:,1])), constr)
prob.solve(method = 'bcd', solver = 'SCS', rho = 1.2, random_ini = 0, max_iter = 200)

time = np.linspace(0,10,100)
plt.plot(np.array(x[0,:].value).flatten(),'g--',linewidth=2)
plt.plot(np.array(x[1,:].value).flatten(),'b-', linewidth=2)
plt.plot(np.array(u.value).flatten(),'r-.', linewidth=2)
plt.legend(["armature current", "rotation speed", "field current"], loc = 4)
plt.show()
