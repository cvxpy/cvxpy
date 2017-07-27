#!/usr/bin/env python3
# @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
# Boyd and Vandenberghe, Convex Optimization, exercise 4.62 page 210

import numpy as np
import cvxpy as cvx

'''
Input parameters   
  n: number of receivers
  a_val: Positive bit rate coefficient for each receiver
  b_val: Positive signal to noise ratio cofficient for each receiver
  P_tot: Total power available to all channels
  W_tot: Total bandwidth available to all channels
'''

def optimal_power(n, a_val, b_val, P_tot=1.0, W_tot=1.0):
  '''
Boyd and Vandenberghe, Convex Optimization, exercise 4.62 page 210
Optimal power and bandwidth allocation in a Gaussian broadcast channel.

We consider a communication system in which a central node transmits messages
to n receivers. Each receiver channel is characterized by its (transmit) power
level Pi ≥ 0 and its bandwidth Wi ≥ 0. The power and bandwidth of a receiver
channel determine its bit rate Ri (the rate at which information can be sent)
via
   Ri=αiWi log(1 + βiPi/Wi),
where αi and βi are known positive constants. For Wi=0, we take Ri=0 (which
is what you get if you take the limit as Wi → 0).  The powers must satisfy a
total power constraint, which has the form
P1 + · · · + Pn = Ptot,
where Ptot > 0 is a given total power available to allocate among the channels.
Similarly, the bandwidths must satisfy
W1 + · · · +Wn = Wtot,
where Wtot > 0 is the (given) total available bandwidth. The optimization
variables in this problem are the powers and bandwidths, i.e.,
P1, . . . , Pn, W1, . . . ,Wn.
The objective is to maximize the total utility, sum(ui(Ri),i=1..n)
where ui: R → R is the utility function associated with the ith receiver.
  '''
  # Input parameters: alpha and beta are constants from R_i equation
  n=len(a_val)
  if n!=len(b_val):
    print('alpha and beta vectors must have same length!')
    return 'failed',np.nan,np.nan,np.nan
  P=cvx.Variable(n)
  W=cvx.Variable(n)
  alpha=cvx.Parameter(n)
  beta =cvx.Parameter(n)
  alpha.value=np.array(a_val)
  beta.value =np.array(b_val)
  # This function will be used as the objective so must be DCP; i.e. element-wise multiplication must occur inside kl_div, not outside otherwise the solver does not know if it is DCP...
  R=cvx.kl_div(cvx.mul_elemwise(alpha, W),
               cvx.mul_elemwise(alpha, W + cvx.mul_elemwise(beta, P))) - \
    cvx.mul_elemwise(alpha, cvx.mul_elemwise(beta, P))
  objective=cvx.Minimize(cvx.sum_entries(R))
  constraints=[P>=0.0,
               W>=0.0,
               cvx.sum_entries(P)-P_tot==0.0,
               cvx.sum_entries(W)-W_tot==0.0]
  prob=cvx.Problem(objective, constraints)
  prob.solve()
  return prob.status,-prob.value,P.value,W.value

if __name__ == '__main__':
  print(optimal_power.__doc__)
  # print all arrays to have 3 significant figures after the decimal place
  np.set_printoptions(precision=3)
  n=5               # number of receivers in the system
  a_val=np.arange(10,n+10)/(1.0*n)  # alpha
  b_val=[1.0]*n  #  beta
  b_val=np.arange(10,n+10)/(1.0*n)  # beta
  P_tot=0.5
  W_tot=1.0
  print('Test problem data:')
  print('n = %d Ptot = %.3f Wtot = %.3f'%(n,P_tot,W_tot,))
  print('α =',a_val)
  print('β =',b_val)
  status,utility,power,bandwidth=optimal_power(n,a_val,b_val,P_tot,W_tot)
  print('Status =',status)
  print('Optimal utility value = %.4g '%utility)
  print('Optimal power level:\n', power)
  print('Optimal bandwidth:\n', bandwidth)
