__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

m = 10
n = 30
k = 5
np.random.seed(0)
A = np.dot(np.abs(np.random.randn(m,k)),np.abs(np.random.randn(k,n)))
U, Sigma, V = np.linalg.svd(A)

#r = [5,10,15,20,25,30,35,40,45,50]
r = [1,2,3,4,5,6,7,8,9,10]
appro_tol = []
appro_tol_lin = []
appro_tol_ran = []
for rank in r:
    X = Variable(m,rank)
    X.value = abs(np.dot(U[:,0:rank],np.diag(np.sqrt(Sigma[0:rank])))).value
    #X.value = np.ones((m,rank))
    Y = Variable(rank,n)
    Y.value = abs(np.dot(np.diag(np.sqrt(Sigma[0:rank])),V[0:rank,:])).value
    #Y.value = np.ones((rank,n))
    obj = Minimize(square(norm(X*Y-A,'fro')))
    prob = Problem(obj, [X>=0, Y>=0])
    #prob.solve(method = 'bcd', solver = 'SCS', random_ini=False)
    print "======= solution ======="
    appro_tol.append(prob.objective.args[0].value/np.linalg.norm(A,'fro'))
    #print "tolerance = ", appro_tol[-1]

    # linear
    #Y.value = np.dot(np.diag(np.sqrt(Sigma[0:rank])),V[0:rank,:])
    #X.value = np.dot(U[:,0:rank],np.diag(np.sqrt(Sigma[0:rank])))
    #prob.solve(method = 'bcd', solver = 'SCS', linear = True, random_ini = 0)
    print "======= solution ======="
    appro_tol_lin.append(prob.objective.args[0].value/np.linalg.norm(A,'fro'))
    #print "tolerance = ", appro_tol_lin[-1]

    # random initial
    prob.solve(method = 'bcd', solver = 'SCS', random_times = 5)
    print "======= solution ======="
    appro_tol_ran.append(prob.objective.args[0].value/np.linalg.norm(A,'fro'))
    print "tolerance = ", appro_tol_ran[-1]

print Sigma
#plt.semilogy(r,appro_tol,'b-o')
#plt.semilogy(r,appro_tol_lin, 'r--^')
plt.semilogy(r,appro_tol_ran, 'b-o')
plt.xlabel("$r$")
plt.ylabel("$||XY-A||_F/||A||_F$")
#plt.legend(["proximal operator with specified initial point", "proximal gradient with specified initial point", "proximal operator with random initial point"])
plt.show()

