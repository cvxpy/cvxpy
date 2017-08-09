# data and code for multiperiod portfolio rebalancing problem
import numpy as np
import time

ANSWERS = []
TIME = 0
T = 100
n = 5
gamma = 8.0
threshold = 0.001
Sigma = np.array([[  1.512e-02,  1.249e-03,  2.762e-04, -5.333e-03, -7.938e-04],
 [  1.249e-03,  1.030e-02,  6.740e-05, -1.301e-03, -1.937e-04],
 [  2.762e-04,  6.740e-05,  1.001e-02, -2.877e-04, -4.283e-05],
 [ -5.333e-03, -1.301e-03, -2.877e-04,  1.556e-02,  8.271e-04],
 [ -7.938e-04, -1.937e-04, -4.283e-05,  8.271e-04,  1.012e-02]])
mug = np.asarray([ 1.02 , 1.028, 1.01 , 1.034, 1.017])
mu = np.asmatrix([ 1.02 , 1.028, 1.01 , 1.034, 1.017])
kappa_1 = np.matrix([ 0.002, 0.002, 0.002, 0.002, 0.002])
kappa_2 = np.matrix([ 0.004, 0.004, 0.004, 0.004, 0.004])

## Generate returns
# call this function to generate a vector r of market returns
generateReturns = lambda: np.random.multivariate_normal(mug,Sigma)


from cvxpy import *
import copy

#Getting w_star
kappa = kappa_2

w = Variable(n)
obj = Maximize(mu * w  - gamma/2 * quad_form(w,Sigma))
constraints = [sum(w) == 1]
prob = Problem(obj, constraints)
tic = time.time()
ANSWERS.append(prob.solve())
toc = time.time()
TIME += toc - tic
w_star = w.value
ws = []
us = []

w_last = w_star
wt = Parameter(n)
w = Variable(n)
obj = Maximize(mu * w  - gamma/2 * quad_form(w,Sigma) - kappa * abs(w - wt)) 
constraints = [sum(w) == 1]
problem = Problem(obj, constraints)
for i in range(T):
	returns = generateReturns()
	wt.value = np.diag(returns) * w_last / ( returns.T *  w_last )
	EPS=1e-4
	tic = time.time()
	ANSWERS.append(problem.solve(solver="ECOS"))
	toc = time.time()
	TIME += toc - tic
	us.append(w.value - wt.value)
	w_last = w.value
	ws.append(w_last)


ws = np.asarray(ws)
us = np.asarray(us)
w_star = np.asarray(w_star)

ws = ws[:,:,0]
us = us[:,:,0]
w_star = w_star[:,0]



## Plotting code
# You must provide three objects:
# - ws: np.array of size T x n,
#       the post-trade weights w_t_tilde;
# - us: np.array of size T x n, 
#       the trades at each period: w_t_tilde - w_t;
# - w_star: np.array of size n,
#       the "target" solution w_star.
import matplotlib.pyplot as plt
colors = ['b','r','g','c','m']
plt.figure(figsize=(13,5))
for j in range(n):
    plt.plot(range(T), ws[:,j], colors[j])
    plt.plot(range(T), [w_star[j]]*T,  colors[j]+'--')
    non_zero_trades = abs(us[:,j]) > threshold
    print non_zero_trades
plt.ylabel('post-trade weights')
plt.xlabel('period $t$')

num_non_zero_trades = 0.0
for i in range(len(us)):
	moved = [us[i,j] > 1e-3 for j in range(len(us[0]))]
	if any(moved):
		num_non_zero_trades +=1

print num_non_zero_trades/(T)

plt.show()
