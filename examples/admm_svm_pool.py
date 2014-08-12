# A SVM example with CVXPY.
from cvxpy import *
import numpy as np
from multiprocessing import Pool
import time

# Divide the data into NUM_PROCS segments,
# using NUM_PROCS processes.
NUM_PROCS = 4
SPLIT_SIZE = 1000

# Problem data.
np.random.seed(1)
N = NUM_PROCS*SPLIT_SIZE
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
rho = 1.0
w = Variable(n + 1)

def prox(args):
    data_split, w_avg = args
    slack = [pos(1 - l*(a.T*w[:-1] - w[-1])) for (l, a) in data_split]
    obj = norm(w, 2) + sum(slack)
    obj += (rho/2)*sum_squares(w - w_avg)
    Problem(Minimize(obj)).solve()
    return w.value

# ADMM algorithm.
pool = Pool(NUM_PROCS)
w_avg = np.random.randn(n+1, 1)
u_vals = NUM_SPLITS*[np.zeros((n+1, 1))]
for i in range(5):
    print get_error(w_avg)
    prox_args = [w_avg - ui for ui in u_vals]
    w_vals = pool.map(prox, zip(data_splits, prox_args))
    w_avg = sum(w_vals)/len(w_vals)
    u_vals = [ui + wi - w_avg for ui, wi in zip(u_vals, w_vals)]

print w_avg[:-1]
print w_avg[-1]
