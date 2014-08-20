#!/usr/bin/env python
# Keith Briggs 2014-08-19
# Fastest-mixing Markov chain
# Boyd, Diaconis, and Xiao SIAM Rev. 46 (2004) 667-689 at p.672

import numpy as np
import cvxpy

def antiadjacency(g):
  # form the complementary graph
  n=1+max(g.keys()) # assumes keys start at 0
  a=dict((i,[]) for i in range(n))
  for x in g:
    for y in range(n):
      if y not in g[x]:
        a[x].append(y)
  return a

def FMMC(g,verbose=False):
  # Fastest-mixing Markov chain on the graph g
  # this is formulation (5), p.672
  # Boyd, Diaconis, and Xiao SIAM Rev. 46 (2004) 667-689
  a=antiadjacency(g)
  n=len(a.keys())
  P=cvxpy.Variable(n,n)
  o=np.ones(n)
  objective=cvxpy.Minimize(cvxpy.norm(P-1.0/n))
  constraints=[P*o==o,P.T==P,P>=0]
  for i in a:
    for j in a[i]: # i-j is a not-edge of g!
      if i!=j: constraints.append(P[i,j]==0)
  prob=cvxpy.Problem(objective,constraints)
  prob.solve()
  if verbose: print 'status: %s.'%prob.status,'optimal value=%.6f'%prob.value
  return prob.status,prob.value,P.value

def print_result(P,n,eps=1e-8):
  for row in P:
    for i in range(n):
      x=row[0,i]
      if abs(x)<eps: x=0.0
      print '%8.4f'%x,
    print

def examples_p674():
  print 'SIAM Rev. 46 examples p.674: Figure 1 and Table 1'
  print '(a) line graph L(4)'
  g={0:(1,),1:(0,2,),2:(1,3,),3:(2,)}
  status,value,P=FMMC(g,verbose=True)
  print_result(P,len(g))
  print '(b) triangle+one edge'
  g={0:(1,),1:(0,2,3,),2:(1,3,),3:(1,2,)}
  status,value,P=FMMC(g,verbose=True)
  print_result(P,len(g))
  print '(c) bipartite 2+3'
  g={0:(1,3,4,),1:(0,2,),2:(1,3,4,),3:(0,2,),4:(0,2,)}
  status,value,P=FMMC(g,verbose=True)
  print_result(P,len(g))
  print '(d) square+central point'
  g={0:(1,2,4,),1:(0,3,4,),2:(0,3,4,),3:(1,2,4,),4:(0,1,2,3,4,)}
  status,value,P=FMMC(g,verbose=True)
  print_result(P,len(g))

if __name__=='__main__':
  examples_p674()
