
Semidefinite program
====================

A semidefinite program (SDP) is an optimization problem of the form

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & \mathbf{tr}(CX) \\
       \mbox{subject to} & \mathbf{tr}(A_iX) = b_i, \quad i=1,\ldots,p \\
                         & X \succeq 0,
       \end{array}

where :math:`\mathbf{tr}` is the trace function,
:math:`X \in \mathcal{S}^{n}` is the optimization variable and
:math:`C, A_1, \ldots, A_p \in \mathcal{S}^{n}`, and
:math:`b_1, \ldots, b_p \in \mathcal{R}` are problem data, and
:math:`X \succeq 0` is a matrix inequality. Here :math:`\mathcal{S}^{n}`
denotes the set of :math:`n`-by-:math:`n` symmetric matrices.

An example of an SDP is to complete a covariance matrix
:math:`\tilde \Sigma \in \mathcal{S}^{n}_+` with missing entries
:math:`M \subset \{1,\ldots,n\} \times \{1,\ldots,n\}`:

.. math::

     
       \begin{array}{ll}
       \mbox{minimize}   & \mathbf{tr}(\Sigma) \\
       \mbox{subject to} & \Sigma_{ij} = \tilde \Sigma_{ij}, \quad (i,j) \notin M \\
                         & \Sigma \succeq 0,
       \end{array}

Example
-------

In the following code, we solve a SDP with CVXPY.

.. code:: python

    # Import packages.
    import cvxpy as cp
    import numpy as np
    
    # Generate a random SDP.
    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())
    
    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                      constraints)
    prob.solve()
    
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)


.. parsed-literal::

    The optimal value is 2.654348003008652
    A solution X is
    [[ 1.6080571  -0.59770202 -0.69575904]
     [-0.59770202  0.22228637  0.24689205]
     [-0.69575904  0.24689205  1.39679396]]

