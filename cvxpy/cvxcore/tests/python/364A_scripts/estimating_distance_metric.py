import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import copy



from scipy import linalg as la
np.random.seed(8)

n = 5 # Dimension
N = 100 # Number of sample
N_test = 10 # Samples for test set

X = np.random.randn(n,N)
Y = np.random.randn(n,N)

X_test = np.random.randn(n,N_test)
Y_test = np.random.randn(n,N_test)

P = np.random.randn(n,n)
P = P.dot(P.T) + np.identity(n)
sqrtP = la.sqrtm(P)

d = np.linalg.norm(sqrtP.dot(X-Y),axis=0)
d = np.maximum(d+np.random.randn(N),0)

d_test = np.linalg.norm(sqrtP.dot(X_test-Y_test),axis=0)
d_test = np.maximum(d_test+np.random.randn(N_test),0)



Z = X - Y  


P = Semidef(n)
objective = quad_form( Z[:,0] ,P )**2  + d[0] **2 -  2*d[0]* sqrt(quad_form(Z[:,0],P)  )
for i in range(1,N):
	objective = quad_form(Z[:,i],P )**2  + d[i] **2 -2*d[i]* sqrt(quad_form(Z[:,i],P) )

#objective /= float(N)

obj = Minimize(objective)
prob = Problem(obj, [])
val = prob.solve()

pass #print "P", P.value
pass #print "training error", val


testing_error = 0

Z_test = X_test - Y_test
for i in range(N_test):
	testing_error += (d_test[i] -  Z_test[:,i].T.dot( P.value.dot(Z_test[:,i]).T))**2

pass #print "Testing error:", testing_error