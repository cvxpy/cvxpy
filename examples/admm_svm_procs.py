# A SVM example with CVXPY.
from cvxpy import *
import numpy as np
from multiprocessing import Process, Pipe
import time

# Divide the data into NUM_PROCS segments,
# using NUM_PROCS processes.
NUM_PROCS = 4
SPLIT_SIZE = 500
MAX_ITER = 20

# Problem data.
np.random.seed(1)
N = NUM_PROCS*SPLIT_SIZE
n = 11
data = []
offset = np.random.randn(n-1, 1)
for i in xrange(N/2):
    data += [(1, offset + np.random.normal(1.0, 4.0, (n-1, 1)))]
for i in xrange(N/2):
    data += [(-1, offset + np.random.normal(-1.0, 4.0, (n-1, 1)))]
np.random.shuffle(data)
data_splits = [data[i:i+SPLIT_SIZE] for i in xrange(0, N, SPLIT_SIZE)]

# Count misclassifications.
def get_error(x):
    error = 0
    for label, sample in data:
        if not label*(np.dot(x[:-1].T, sample) - x[-1])[0] > 0:
            error += 1
    return "%d misclassifications out of %d samples" % (error, N)

# Construct problem.
gamma = 0.1
rho = 1.0
x = Variable(n)
f = []
for split in data_splits:
    slack = [pos(1 - b*(a.T*x[:-1] - x[-1])) for (b, a) in split]
    f += [norm(x, 2) + gamma*sum(slack)]

# Process:
# Send xi, wait for xbar
def run_process(f, pipe):
    xbar = Parameter(n, value=np.zeros(n))
    u = Parameter(n, value=np.zeros(n))
    f += (rho/2)*sum_squares(x - xbar + u)
    prox = Problem(Minimize(f))
    # ADMM loop.
    while True:
        prox.solve()
        pipe.send(x.value)
        xbar.value = pipe.recv()
        u.value += x.value - xbar.value

# Setup.
pipes = []
procs = []
for i in range(NUM_PROCS):
    local, remote = Pipe()
    pipes += [local]
    procs += [Process(target=run_process, args=(f[i], remote))]
    procs[-1].start()

# ADMM loop.
for i in range(MAX_ITER):
    # Gather.
    xbar = sum([pipe.recv() for pipe in pipes])/NUM_PROCS
    print get_error(xbar)
    # Scatter.
    [pipe.send(xbar) for pipe in pipes]

[p.terminate() for p in procs]
# # Best solution.
# start = time.time()
# slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data]
# obj = norm(w, 2) + gamma*sum(slack)
# Problem(Minimize(obj)).solve()
# print "Time elapsed =", time.time() - start

# print "Standard solution:", get_error(w.value)

# def prox(args):
#     data_split, w_avg = args
#     slack = [pos(1 - b*(a.T*w[:-1] - w[-1])) for (b, a) in data_split]
#     obj = norm(w, 2) + gamma*sum(slack)
#     obj += (rho/2)*sum_squares(w - w_avg)
#     Problem(Minimize(obj)).solve(solver=SCS)
#     return w.value

# # ADMM algorithm.
# pool = Pool(NUM_PROCS)
# w_avg = np.random.randn(n+1, 1)
# u_vals = NUM_PROCS*[np.zeros((n+1, 1))]
# start = time.time()
# for i in range(5):
#     print get_error(w_avg)
#     prox_args = [w_avg - ui for ui in u_vals]
#     w_vals = pool.map(prox, zip(data_splits, prox_args))
#     w_avg = sum(w_vals)/len(w_vals)
#     u_vals = [ui + wi - w_avg for ui, wi in zip(u_vals, w_vals)]
# print "Time elapsed =", time.time() - start

# print w_avg[:-1]
# print w_avg[-1]

