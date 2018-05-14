from cvxpy import *
import numpy
import time


ANSWERS = []
TIME = 0
# SVM with indexing.
def gen_data(n):
    pos = numpy.random.multivariate_normal([1.0,2.0],numpy.eye(2),size=n)
    neg = numpy.random.multivariate_normal([-1.0,1.0],numpy.eye(2),size=n)
    return pos, neg

N = 2
C = 10
pos, neg = gen_data(500)

w = Variable(N)
b = Variable()
xi_pos = Variable(pos.shape[0])
xi_neg = Variable(neg.shape[0])
cost = sum_squares(w) + C*sum_entries(xi_pos) + C*sum_entries(xi_neg)
constrs = []
for j in range(pos.shape[0]):
    constrs += [w.T*pos[j,:] - b >= 1 - xi_pos[j]]
    
for j in range(neg.shape[0]):
    constrs += [-(w.T*neg[j,:] - b) >= 1 - xi_neg[j]]

p = Problem(Minimize(cost), constrs)

tic =  time.time()
ANSWERS.append(p.solve())
toc = time.time()
TIME += toc - tic