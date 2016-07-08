__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

m = 10
n = 10
T = 10
times = 1
alpha_value = np.logspace(-5,0,50)
Error = np.zeros((1,len(alpha_value)))
Cardinality = np.zeros((1,len(alpha_value)))
Error_random = np.zeros((1,len(alpha_value)))
Cardinality_random = np.zeros((1,len(alpha_value)))

for t in range(times):
    X = np.random.randn(m,T)
    U, s, V = np.linalg.svd(X)

    D = Variable(m,n)
    D.value = np.dot(U,np.diag(s))
    Y = Variable(n,T)
    Y.value = V
    alpha = Parameter(sign = 'Positive')
    cost = square(norm(D*Y-X,'fro'))/2+alpha*norm(Y,1)
    obj = Minimize(cost)
    prob = Problem(obj,[norm(D,'fro')<=1])

    err = []
    card = []
    err_random = []
    card_random = []
    for a_value in alpha_value:
        alpha.value = a_value
        prob.solve(method = 'bcd', solver = 'SCS', ep = 1e-2, mu_max = 1e4, lambd = 10, random_ini = False)
        err.append(norm(D*Y-X,'fro').value/norm(X,'fro').value)
        card.append(sum_entries(abs(Y).value>=1e-3).value)
        print "======= solution ======="
        print "objective =", cost.value
        #### random initial point
        prob.solve(method = 'bcd', solver = 'SCS', ep = 1e-2, mu_max = 1e4, lambd = 10)
        err_random.append(norm(D*Y-X,'fro').value/norm(X,'fro').value)
        card_random.append(sum_entries(abs(Y).value>=1e-3).value)
        print "======= solution ======="
        print "objective =", cost.value
    Error += np.array(err).flatten()/float(times)
    Cardinality += np.array(card).flatten()/float(times)
    Error_random += np.array(err_random).flatten()/float(times)
    Cardinality_random += np.array(card_random).flatten()/float(times)

plt.plot(Cardinality, Error, 'b o')
plt.plot(Cardinality_random, Error_random, 'r s')
plt.xlabel('Cardinality of $Y$')
plt.ylabel('$||D*Y-X||_F/||X||_F$')
#plt.legend(["initial points from svd","random initial points"])
plt.show()
