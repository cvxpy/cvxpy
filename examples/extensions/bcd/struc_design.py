__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 9 # number of bars
F = [1]*n # external forces
beta = 10
l0 = 8 # total length

M = np.eye(n)
M_ind = np.tril_indices(n)
M[M_ind] = 1

a = Variable(n) # width
l = Variable(n) # length

l.value = np.ones((n,1))
a.value = np.ones((n,1))

cost = a.T*l
constr = [a>=0, l>=0, beta*a >= M*(F+diag(l)*a), sum_entries(l) == l0]
prob = Problem(Minimize(cost), constr)
iter, max_slack = bcd(prob, solver = 'SCS', linear=False, lambd = 1, random_ini = 0, ep = 1e-4, random_times = 1, mu = 5)
print "======= solution ======="
print "number of iterations =", iter+1
print "objective =", cost.value
print a.value
print l.value

# plot
rear = 0
for i in range(n):
    head = rear
    rear = head+l[i].value
    width = a[i].value*10
    plt.plot([0,0],[head,rear],'b-',linewidth=width,solid_capstyle="butt")

plt.plot([-0.3,0.3],[l0,l0],'k-',solid_capstyle="butt")
plt.plot([0,0.1],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.plot([0.1,0.2],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.plot([0.2,0.3],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.plot([-0.1,0],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.plot([-0.2,-0.1],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.plot([-0.3,-0.2],[l0,l0+0.2],'k-',solid_capstyle="butt")
plt.grid()

plt.xlim([-0.5,0.5])
plt.ylim([-0.5,l0+0.5])
plt.show()

"""
N = 5 # number of fixed nodes
k = N*(N-1) # number of free nodes

P0 = [[0,1,2,3,4]*5, [4]*5 + [3]*5 + [2]*5 + [1]*5 + [0]*5]

m = 25 # total number of nodes
r = []
s = []

dx = [-1, 0, 1, 1]
dy = [-1, -1, -1, 0]
for ind in range(m):
    for i in range(4):
        px = P0[0][ind] + dx[i]
        py = P0[1][ind] + dy[i]
        if 0 <= px and px < N and 0 <= py and py < N:
            r.append(ind)

s = [6,7,2,6,7,8,3,7,8,9,4,8,9,10,5,9,10,11,12,7,11,12,13,8,12,13,14,9,13,14,15,10,14,15,16,17,12,16,17,18,13,17,18,19,14,18,19,20,15,19,20,21,22,17,21,22,23,18,22,23,24,19,23,24,25,20,24,25,22,23,24,25]
s = np.array(s)-1

n = len(r) # number of bars
sigma = 1

M = 1 # number of loadings
F = np.zeros((2, k, 4))
# load 1: (nominal) equal forces down
F[1, :, 0] = -2

# load 2/3: nominal plus forces down-right and down-left
F[0, :, 1] =  np.random.rand(1, k)
F[0, :, 2] = -np.random.rand(1, k)
F[1, :, 1] = F[1, :, 0] - np.random.rand(1, k)
F[1, :, 2] = F[1, :, 0] - np.random.rand(1, k)

# load 4: nominal plus random forces in arbitrary directions
F[:, :, 3] = F[:, :, 0] + np.random.randn(2, k)

A = np.zeros((m, n))
for i in range(n):
    A[r[i], i] = +1
    A[s[i], i] = -1
a = Variable(n)
t = Variable(n,M)
P = Variable(2,m)

a.value = np.ones((n,1))
t.value = np.ones((n,M))
P.value = np.array(P0)

cost = norm2(P*A, axis = 0)*a

constr = [P[:,0:5] == np.array(P0)[:,0:5], P[:,m-5:m] == np.array(P0)[:,m-5:m], a>=0]
#constr = [P == np.array(P0), a>=0]
for i in range(M):
    constr.append(abs(t[:, i]) <= sigma*a)
    constr.append(-P*A*diag(t[:,i])*A.T[:,0:k] + F[:, :, i] == 0)
prob = Problem(Minimize(cost), constr)
iter, max_slack = bcd(prob, solver = 'SCS', linear=False, lambd = 10, max_iter = 200, random_ini = 0, random_times = 1, mu=0.1)
print "======= solution ======="
print "number of iterations =", iter+1
print "objective =", cost.value

# plot
for i in range(n):
    p1 = r[i]
    p2 = s[i]
    plt_str = 'b-'
    width = a[i].value
    if a[i].value < 0.1:
        plt_str = 'r--'
        width = 1
    plt.plot([P[0, p1].value,P[0, p2].value], [P[1, p1].value, P[1, p2].value],plt_str,linewidth=width)
plt.xlim([-0.5,4.5])
plt.ylim([-0.5,4.5])
plt.show()
"""