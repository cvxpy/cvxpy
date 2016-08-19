__author__ = 'Xinyue'

from cvxpy import *
from bcd import bcd
import numpy as np
import matplotlib.pyplot as plt

m = 10
n = 20
T = 20
times = 1
alpha_value = np.logspace(-5,0,50)

Error_random = np.zeros((1,len(alpha_value)))
Cardinality_random = np.zeros((1,len(alpha_value)))

for t in range(times):
    X = np.random.randn(m,T)

    D = Variable(m,n)
    Y = Variable(n,T)
    alpha = Parameter(sign = 'Positive')
    cost = square(norm(D*Y-X,'fro'))/2+alpha*norm(Y,1)
    obj = Minimize(cost)
    prob = Problem(obj,[norm(D,'fro')<=1])

    err = []
    card = []
    err_random = []
    card_random = []
    for a_value in alpha_value:
        D.value = None
        Y.value = None
        alpha.value = a_value
        prob.solve(method = 'bcd', ep = 1e-3, lambd = 200, rho = 1.2, max_iter = 200)
        err_random.append(norm(D*Y-X,'fro').value/norm(X,'fro').value)
        card_random.append(sum_entries(abs(Y).value>=1e-3).value)
        print "======= solution ======="
        print "objective =", cost.value
    Error_random += np.array(err_random).flatten()/float(times)
    Cardinality_random += np.array(card_random).flatten()/float(times)

plt.plot(Cardinality_random, Error_random, 'b o')
plt.xlabel('Cardinality of $Y$')
plt.ylabel('$||D*Y-X||_F/||X||_F$')
plt.show()
