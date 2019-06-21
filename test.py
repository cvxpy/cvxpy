import cvxpy as cp
import numpy as np
input_size = 3
num_groups =2
output_size = 2
lambdas_f = cp.Parameter(nonneg=True)
lambdas_f.value = 30
# Define problem
x_v_f = cp.Variable(input_size*num_groups)
p_f = cp.Variable(1)
q_f = cp.Variable(1)

G = np.zeros((input_size,num_groups,output_size))
a_f = []
d_f = []

for ii in range(input_size):
    G[ii,:,:] = (F@A[:,ii:input_size*num_groups:input_size]).transpose()
for ii in range(input_size):
    a_f.append(cp.norm(x_v_f[ii:input_size*num_groups:input_size],2))
for ii in range(input_size):
    d_f.append(cp.norm(G[ii]@(y-F@A@x_v_f),2))
objective_f = 0.5*p**2+lambdas*q
constr_f = [cp.norm(d_f,"Inf") <= p, sum(a) <= q]
prob_f = cp.Problem(cp.Minimize(objective_f), constr_f)
prob_f.solve()
