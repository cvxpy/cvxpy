import numpy as np
import cvxpy as cp
def random_HPD(n):
  P=np.random.randn(n,n)+1j*np.random.randn(n,n)
  return np.conjugate(P.T)@P
 
def test_02(n=10):
  A=random_HPD(n)
  B=random_HPD(n)
  x=cp.Variable(n,complex=True, name='x')
  obj=cp.gen_lambda_max(A,B)
  prob=cp.Problem(cp.Maximize(obj))
  assert prob.is_dqcp()
  prob.solve(qcp=True,solver=cp.SCS,verbose=True)
  print('optimal value=',prob.value)
  print('x.value=',x.value)

if __name__=='__main__':
  np.random.seed(1)
  print ('cvxpy.__version__=',cp.__version__)
  print('installed_solvers:',cp.installed_solvers())
  test_02()
