#!/usr/bin/env python3
# @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
# Boyd and Vandenberghe, Convex Optimization, exercise 4.20 page 196

import cvxpy as cvx
import numpy as np

'''
Input parameters
  G: matrix of path gains from transmitters to receivers
  P_max: Maximum power that can be transmitted
  P_received: Maximum power that can be received
  sigma: Noise at each receiver
  Group: Power supply groups the transmitters belong to
  Group_max: The maximum power that can be supplied to each group
  detail: Detailed output information is printed if True
  epsilon: Level of precision for the bisection algorithm
'''

def maxmin_sinr(G,P_max,P_received,sigma,Group,Group_max,detail=False,epsilon = 0.001):
  '''
Boyd and Vandenberghe, Convex Optimization, exercise 4.20 page 196
Power assignment in a wireless communication system.
  
We consider n transmitters with powers p1,...,pn ≥ 0, transmitting to
n receivers. These powers are the optimization variables in the problem.
We let G ∈ ℝ(n*n) denote the matrix of path gains from the
transmitter to the receiver. Signal is defined as G_(i,i)*P_i, and
interference is defined as ∑(G_(i,j)*pj). Then signal to interference plus
noise ratio is defined as S_i/(I_i+σ). The objective function is then to
maximise the minimum SINR for all receivers. Each transmitter must be below
a given threshold P_max.  Furthermore, the transmitters are partitioned
into groups, with each group sharing the same power supply.  Therefore there
is a power constraint for each group of transmitter powers.
The receivers have the constraint that they cannot receiver more than
a given amount of power i.e. a saturation threshold.
  '''
  # find n and m from the size of the path gain matrix
  n,m = np.shape(G)
  # Checks sizes of inputs
  if m != np.size(P_max):
    print('Error: P_max dimensions do not match gain matrix dimensions\n')
    return 'Error: P_max dimensions do not match gain matrix dimensions\n',np.nan,np.nan,np.nan

  if n != np.size(P_received):
    print('Error: P_received dimensions do not match gain matrix dimensions\n')
    return 'Error: P_received dimensions do not match gain matrix dimensions',np.nan,np.nan,np.nan

  if n != np.size(sigma):
    print('Error: σ dimensions do not match gain matrix dimensions\n')
    return 'Error: σ dimensions do not match gain matrix dimensions',np.nan,np.nan,np.nan

  I = np.zeros((n,m))
  S = np.zeros((n,m))
  delta = np.identity(n)
  S = G*delta # signal power matrix
  I = G-S # interference power matrix
  # group matrix: number of groups by number of transmitters
  num_groups = int(np.size(Group,0))

  if num_groups != np.size(Group_max):
    print('Error: Number of groups from Group matrix does not match dimensions of Group_max\n')
    return 'Error: Number of groups from Group matrix does not match dimensions of Group_max',np.nan,np.nan,np.nan,np.nan

  # normalising the max power of a group so it is in the range [0,1]
  Group_norm = Group/np.sum(Group,axis=1).reshape((num_groups,1))
  # create scalar optimisation variable p: the power of the n transmitters
  p = cvx.Variable(n)
  best = np.zeros(n)
  # set upper and lower bounds for sub-level set
  u = 1e4
  l = 0
  # alpha defines the sub-level sets of the generalised linear fractional problem
  # in this case alpha is the reciprocal of the minimum SINR
  alpha = cvx.Parameter(rows=1,cols=1)
  # set up the constraints for the bisection feasibility test
  constraints = [I*p + sigma <= alpha*S*p, p <= P_max, p >= 0, G*p <= P_received, Group_norm*p <= Group_max]

  # define objective function, in our case it's constant as only want to test the solution's feasibility
  obj = cvx.Minimize(alpha)
  # now check whether the solution lies between u and l
  alpha.value = u
  prob = cvx.Problem(obj, constraints)
  prob.solve()
  if prob.status != 'optimal':
    # in this case the level set u is below the solution
    print('No optimal solution within bounds\n')
    return 'Error: no optimal solution within bounds',np.nan,np.nan,np.nan

  alpha.value = l
  prob = cvx.Problem(obj, constraints)
  prob.solve()
  if prob.status == 'optimal':
    # in this case the level set l is below the solution
    print('No optimal solution within bounds\n')
    return 'Error: no optimal solution within bounds',np.nan,np.nan,np.nan

  # Bisection algortithm starts
  maxLoop = int(1e7)
  for i in range(1,maxLoop):
    # First check that u is in the feasible domain and l is not, loop finishes here if this is not the case
    # set alpha as the midpoint of the interval
    alpha.value = (u + l)/2.0
    # test the size of the interval against the specified tolerance
    if u-l <= epsilon:
      break

    # form and solve problem
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    # If the problem is feasible u -> alpha, if not l -> alpha, best takes the last feasible value as the optimal one as
    # when the tolerance is reached the new alpha may be out of bounds
    if prob.status == 'optimal':
      u = alpha.value
      best = p.value
    else:
      l = alpha.value

    # final condition to check that the interval has converged to order epsilon, i.e. the range of the optimal sublevel set is <=epsilon
    if u - l > epsilon and i == (maxLoop-1):
      print("Solution not converged to order epsilon")

  # print out various details of solution
  if detail :
    print('l = ', l)
    print('u = ', u)
    print('α = ', alpha.value)
    print('Optimal power p = \n', best)
    print('Received power G*p = \n', (G*best))

  return l,u,alpha.value,best

# this section shows a simple example with 5 transmitters and 5 receivers
if __name__=='__main__':
  print(maxmin_sinr.__doc__)
  # print all arrays to have 3 significant figures after the decimal place
  np.set_printoptions(precision=3)
  # in this case we will use a gain matrix with a signal weight of 0.6 and an inteference weight of 0.1
  G = np.array([[0.6,0.1,0.1,0.1,0.1],
                [0.1,0.6,0.1,0.1,0.1],
                [0.1,0.1,0.6,0.1,0.1],
                [0.1,0.1,0.1,0.6,0.1],
                [0.1,0.1,0.1,0.1,0.6]])
  # in this case m=n, but this generalises if we want n receivers and m transmitters
  n,m = np.shape(G)
  # set maximum power of each transmitter and receiver saturation level
  P_max = np.array([1.]*n)
  # normalised received power, total possible would be all power from all transmitters so 1/n
  P_received = np.array([4.,4.,4.,4.,4.])/n
  # set noise level
  sigma = np.array([0.1,0.1,0.1,0.1,0.1])
  # group matrix: number of groups by number of transmitters
  Group = np.array([[1.,1.,0,0,0],[0,0,1.,1.,1.]])
  # max normalised power for groups, number of groups by 1
  Group_max = np.array([[1.8],[1.8]])
  # now run the optimisation problem
  print('Test problem data')
  print('G=%s'%G)
  print('P_max=%s'%P_max)
  print('P_received=%s'%P_received)
  print('Grouping=%s'%Group)
  print('Max group output=%s'%Group_max)
  l,u,alpha,best=maxmin_sinr(G,P_max,P_received,sigma,Group,Group_max,detail=False)
  print('Max SINR=%.4g'%(1/alpha))
  print('Power=%s'%(best))
