# A SVM example with CVXPY.
from cvxpy import *
import numpy as np
from multiprocessing import Pool, Process, Value, Array, Semaphore
import time

# Divide the data into NUM_SPLITS segments,
# using NUM_PROCS processes.
NUM_PROCS = 8
NUM_SPLITS = 200
SPLIT_SIZE = 100

# Problem data.
np.random.seed(1)
N = NUM_SPLITS*SPLIT_SIZE
n = 10
data = []
for i in xrange(N/2):
    data += [(1, np.random.normal(1.0, 2.0, (n, 1)))]
for i in xrange(N/2):
    data += [(-1, np.random.normal(-1.0, 2.0, (n, 1)))]
data_splits = [data[i:i+SPLIT_SIZE] for i in xrange(0, N, SPLIT_SIZE)]

# Count misclassifications.
def get_error(w):
    error = 0
    for label, sample in data:
        if not label*(np.dot(w[:-1].T, sample) - w[-1])[0] >= 0:
            error += 1
    return "%d misclassifications out of %d samples" % (error, N)

# Construct problem.
gamma = 0.1
rho = 1.0
w = Variable(n + 1)

# Best solution.
start = time.time()
slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data]
obj = norm(w, 2) + gamma*sum(slack)
Problem(Minimize(obj)).solve()
print "Time elapsed =", time.time() - start

print "Standard solution:", get_error(w.value)

def prox(args):
    data_split, w_avg = args
    slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data_split]
    obj = norm(w, 2) + gamma*sum(slack)
    obj += (rho/2)*sum_squares(w - w_avg)
    Problem(Minimize(obj)).solve()
    return w.value

# ADMM algorithm.
pool = Pool(NUM_PROCS)
w_avg = np.random.randn(n+1, 1)
u_vals = NUM_SPLITS*[np.zeros((n+1, 1))]
start = time.time()
for i in range(5):
    print get_error(w_avg)
    prox_args = [w_avg - ui for ui in u_vals]
    w_vals = pool.map(prox, zip(data_splits, prox_args))
    w_avg = sum(w_vals)/len(w_vals)
    u_vals = [ui + wi - w_avg for ui, wi in zip(u_vals, w_vals)]
print "Time elapsed =", time.time() - start

print w_avg[:-1]
print w_avg[-1]

