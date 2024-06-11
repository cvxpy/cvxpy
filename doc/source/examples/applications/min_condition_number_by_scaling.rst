Minimizing Condition Number by Scaling
======================================

This notebook provides and example of how to minimize the condition
number of a matrix by scaling, taken from
https://web.stanford.edu/~boyd/lmibook/. The problem is formulated in
CVXPY as a generalized eigenvalue problem (GEVP) and utilizes the
`DQCP <https://www.cvxpy.org/tutorial/dqcp/index.html>`__ capabilities
in CVXPY. The problem for a matrix
:math:`M \in \mathcal{R}^{m \times n}`, with :math:`m \ge n` is

.. math::


   \begin{array}{llr}
   \text{minimize}   & \gamma^2 & \\
   \text{subject to} & P \in \mathcal{R}^{m \times m} \text{ and diagonal}, & P > 0, \\
                     & Q \in \mathcal{R}^{n \times n} \text{ and diagonal}, & Q > 0, \\
                     & Q \le M^T P M \le \gamma^2 Q &
   \end{array}

Example
=======

In the following code, we solve the above GEVP with CVXPY.

.. code:: ipython3

    # import packages
    import cvxpy as cp
    import numpy as np
    
    # create helper functions
    def cond(A):
        return np.linalg.cond(A)
    
    def evalCond(M,Q,P):
        L     = np.diag(np.diag(P.value)**(1/2))
        R     = np.diag(np.diag(Q.value)**(-1/2))
        Mnew  = L @ M @ R
        return np.linalg.cond(Mnew)
    
    # create a random matrix
    m = 3
    n = 2
    np.random.seed(2)
    M = np.random.rand(m,n)
    
    # specify the variables
    p = cp.Variable(m,pos=True)
    P = cp.diag(p)
    q = cp.Variable(n,pos=True)
    Q = cp.diag(q)
    
    # define the variables for GEVP
    A = M.T @ P @ M
    B = Q
    C = A - Q
    
    # create the constraints and objective
    ep = 1e-3
    constr = [C >= ep*np.eye(C.shape[0]),
              P >= ep*np.eye(P.shape[0]),
              Q >= ep*np.eye(Q.shape[0])]
    
    # note: the variable lambda = gamma^2 from the problem statement
    objFun = cp.Minimize(cp.gen_lambda_max(A,B))
    
    # create the problem
    problem = cp.Problem(objFun,constr)
    
    # check if DQCP
    print("Is the problem DQCP? ",problem.is_dqcp())
    
    # solve
    problem.solve(qcp=True,solver="SCS")
    
    # print results
    if problem.status not in ["infeasible", "unbounded"]:
        print("Initial Condition Number: ",cond(M))
        print("Optimized Condition Number: ",evalCond(M,Q,P))
    else:
        print(problem.status)


.. parsed-literal::

    Is the problem DQCP?  True
    Initial Condition Number:  4.1538811703979786
    Optimized Condition Number:  1.7548711807791855

