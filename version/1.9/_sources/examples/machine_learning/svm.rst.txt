
Support vector machine classifier with :math:`\ell_1`-regularization
====================================================================

In this example we use CVXPY to train a SVM classifier with
:math:`\ell_1`-regularization. We are given data :math:`(x_i,y_i)`,
:math:`i=1,\ldots, m`. The :math:`x_i \in {\bf R}^n` are feature
vectors, while the :math:`y_i \in \{\pm 1\}` are associated boolean
outcomes. Our goal is to construct a good linear classifier
:math:`\hat y = {\rm sign}(\beta^T x - v)`. We find the parameters
:math:`\beta,v` by minimizing the (convex) function

.. math::


   f(\beta,v) = (1/m) \sum_i \left(1 - y_i ( \beta^T x_i-v) \right)_+ + \lambda
   \| \beta\|_1

The first term is the average hinge loss. The second term shrinks the
coefficients in :math:`\beta` and encourages sparsity. The scalar
:math:`\lambda \geq 0` is a (regularization) parameter. Minimizing
:math:`f(\beta,v)` simultaneously selects features and fits the
classifier.

Example
~~~~~~~

In the following code we generate data with :math:`n=20` features by
randomly choosing :math:`x_i` and a sparse
:math:`\beta_{\mathrm{true}} \in {\bf R}^n`. We then set
:math:`y_i = {\rm sign}(\beta_{\mathrm{true}}^T x_i -v_{\mathrm{true}} - z_i)`,
where the :math:`z_i` are i.i.d. normal random variables. We divide the
data into training and test sets with :math:`m=1000` examples each.

.. code:: python

    # Generate data for SVM classifier with L1 regularization.
    from __future__ import division
    import numpy as np
    np.random.seed(1)
    n = 20
    m = 1000
    TEST = m
    DENSITY = 0.2
    beta_true = np.random.randn(n,1)
    idxs = np.random.choice(range(n), int((1-DENSITY)*n), replace=False)
    for idx in idxs:
        beta_true[idx] = 0
    offset = 0
    sigma = 45
    X = np.random.normal(0, 5, size=(m,n))
    Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0,sigma,size=(m,1)))
    X_test = np.random.normal(0, 5, size=(TEST,n))
    Y_test = np.sign(X_test.dot(beta_true) + offset + np.random.normal(0,sigma,size=(TEST,1)))

We next formulate the optimization problem using CVXPY.

.. code:: python

    # Form SVM with L1 regularization problem.
    import cvxpy as cp
    beta = cp.Variable((n,1))
    v = cp.Variable()
    loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
    reg = cp.norm(beta, 1)
    lambd = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(loss/m + lambd*reg))

We solve the optimization problem for a range of :math:`\lambda` to
compute a trade-off curve. We then plot the train and test error over
the trade-off curve. A reasonable choice of :math:`\lambda` is the value
that minimizes the test error.

.. code:: python

    # Compute a trade-off curve and record train and test error.
    TRIALS = 100
    train_error = np.zeros(TRIALS)
    test_error = np.zeros(TRIALS)
    lambda_vals = np.logspace(-2, 0, TRIALS)
    beta_vals = []
    for i in range(TRIALS):
        lambd.value = lambda_vals[i]
        prob.solve()
        train_error[i] = (np.sign(X.dot(beta_true) + offset) != np.sign(X.dot(beta.value) - v.value)).sum()/m
        test_error[i] = (np.sign(X_test.dot(beta_true) + offset) != np.sign(X_test.dot(beta.value) - v.value)).sum()/TEST
        beta_vals.append(beta.value)

.. code:: python

    # Plot the train and test error over the trade-off curve.
    import matplotlib.pyplot as plt
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'
    
    plt.plot(lambda_vals, train_error, label="Train error")
    plt.plot(lambda_vals, test_error, label="Test error")
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.show()



.. image:: svm_files/svm_8_0.svg


We also plot the regularization path, or the :math:`\beta_i` versus
:math:`\lambda`. Notice that the :math:`\beta_i` do not necessarily
decrease monotonically as :math:`\lambda` increases. 4 features remain
non-zero longer for larger :math:`\lambda` than the rest, which suggests
that these features are the most important. In fact
:math:`\beta_{\mathrm{true}}` had 4 non-zero values.

.. code:: python

    # Plot the regularization path for beta.
    for i in range(n):
        plt.plot(lambda_vals, [wi[i,0] for wi in beta_vals])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")



.. image:: svm_files/svm_10_0.svg

