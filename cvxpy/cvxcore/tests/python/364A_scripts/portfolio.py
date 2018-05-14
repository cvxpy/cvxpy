import numpy as np
from cvxpy import *
import matplotlib.pyplot as pyplot
import time

TIME = 0
ANSWERS = []

Sigma = Semidef(4)

x = np.array([.1, .2, -.05, .1])

constraints = [Sigma == Sigma.T]

constraints += [Sigma[0,1] >= 0, Sigma[0,2] >= 0,\
Sigma[1,2] <= 0, Sigma[2,3] <= 0, Sigma[1,3]<=0,\
Sigma[0,0] == .2, Sigma[1,1] == .1, Sigma[2,2] == .3,\
Sigma[3,3] == .1]


objective = Maximize(quad_form(x,Sigma))

prob = Problem(objective, constraints)

tic = time.time()
risk = prob.solve()
toc = time.time()
TIME += toc - tic
ANSWERS.append(risk)

pass #print "Risk:, ", risk
pass #print "Sigma: ", Sigma.value

pass #print "Diagonal risk: ", x.T.dot(np.diag(np.diag(Sigma.value)).dot(x))