
Perron-Frobenius matrix completion
==================================

The DGP atom library has several functions of positive matrices,
including the trace, (matrix) product, sum, Perron-Frobenius eigenvalue,
and :math:`(I - X)^{-1}` (eye-minus-inverse). In this notebook, we use
some of these atoms to formulate and solve an interesting matrix
completion problem.

In this problem, we are given some entries of an elementwise positive
matrix :math:`A`, and the goal is to choose the missing entries so as to
minimize the Perron-Frobenius eigenvalue or spectral radius. Letting
:math:`\Omega` denote the set of indices :math:`(i, j)` for which
:math:`A_{ij}` is known, the optimization problem is

.. math::


   \begin{equation}
   \begin{array}{ll}
   \mbox{minimize} & \lambda_{\text{pf}}(X) \\
   \mbox{subject to} & \prod_{(i, j) \not\in \Omega} X_{ij} = 1 \\
   & X_{ij} = A_{ij}, \, (i, j) \in \Omega,
   \end{array}
   \end{equation}

which is a log-log convex program. Below is an implementation of this
problem, with specific problem data

.. math::


   A = \begin{bmatrix}
   1.0 & ? &  1.9 \\
   ? & 0.8 &  ? \\
   3.2 & 5.9&  ?
   \end{bmatrix},

where the question marks denote the missing entries.

.. code:: python

    import cvxpy as cp
    
    n = 3
    known_value_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
    known_values = [1.0, 1.9, 0.8, 3.2, 5.9]
    X = cp.Variable((n, n), pos=True)
    objective_fn = cp.pf_eigenvalue(X)
    constraints = [
      X[known_value_indices] == known_values,
      X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
    ]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(gp=True)
    print("Optimal value: ", problem.value)
    print("X:\n", X.value)


.. parsed-literal::

    Optimal value:  4.702374203221372
    X:
     [[1.         4.63616907 1.9       ]
     [0.49991744 0.8        0.37774148]
     [3.2        5.9        1.14221476]]

