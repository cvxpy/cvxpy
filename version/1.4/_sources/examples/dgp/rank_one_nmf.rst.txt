
Rank-one nonnegative matrix factorization
=========================================

The DGP atom library has several functions of positive matrices,
including the trace, (matrix) product, sum, Perron-Frobenius eigenvalue,
and :math:`(I - X)^{-1}` (eye-minus-inverse). In this notebook, we use
some of these atoms to approximate a partially known elementwise
positive matrix as the outer product of two positive vectors.

We would like to approximate :math:`A` as the outer product of two
positive vectors :math:`x` and :math:`y`, with :math:`x` normalized so
that the product of its entries equals :math:`1`. Our criterion is the
average relative deviation between the entries of :math:`A` and
:math:`xy^T`, that is,

.. math::


   \frac{1}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} R(A_{ij}, x_iy_j),

where :math:`R` is the relative deviation of two positive numbers,
defined as

.. math::


   R(a, b) = \max\{a/b, b/a\} - 1.

The corresponding optimization problem is

.. math::


   \begin{equation}
   \begin{array}{ll}
   \mbox{minimize} & \frac{1}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} R(X_{ij}, x_iy_j)
   \\
   \mbox{subject to} & x_1x_2 \cdots x_m = 1 \\
   & X_{ij} = A_{ij}, \quad \text{for } (i, j) \in \Omega,
   \end{array}
   \end{equation}

with variables :math:`X \in \mathbf{R}^{m \times n}_{++}`,
:math:`x \in \mathbf{R}^{m}_{++}`, and
:math:`y \in \mathbf{R}^{n}_{++}`. We can cast this problem as an
equivalent generalized geometric program by discarding the :math:`-1`
from the relative deviations.

The below code constructs and solves this optimization problem, with
specific problem data

.. math::


   A = \begin{bmatrix}
   1.0 & ? &  1.9 \\
   ? & 0.8 &  ? \\
   3.2 & 5.9&  ?
   \end{bmatrix},

.. code:: python

    import cvxpy as cp
    
    m = 3
    n = 3
    X = cp.Variable((m, n), pos=True)
    x = cp.Variable((m,), pos=True)
    y = cp.Variable((n,), pos=True)
    
    outer_product = cp.vstack([x[i] * y for i in range(m)])
    relative_deviations = cp.maximum(
      cp.multiply(X, outer_product ** -1),
      cp.multiply(X ** -1, outer_product))
    objective = cp.sum(relative_deviations)
    constraints = [
      X[0, 0] == 1.0,
      X[0, 2] == 1.9,
      X[1, 1] == 0.8,
      X[2, 0] == 3.2,
      X[2, 1] == 5.9,
      x[0] * x[1] * x[2] == 1.0,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(gp=True)
    
    print("Optimal value:\n", 1.0/(m * n) * (problem.value - m * n), "\n")
    print("Outer product approximation\n", outer_product.value, "\n")
    print("x: ", x.value)
    print("y: ", y.value)


.. parsed-literal::

    Optimal value:
     1.7763568394002505e-14 
    
    Outer product approximation
     [[1.         1.84375    1.9       ]
     [0.43389831 0.8        0.82440678]
     [3.2        5.89999999 6.07999999]] 
    
    x:  [0.89637009 0.38893346 2.86838428]
    y:  [1.11561063 2.0569071  2.1196602 ]

