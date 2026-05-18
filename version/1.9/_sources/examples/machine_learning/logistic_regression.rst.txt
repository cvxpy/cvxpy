
Logistic regression with :math:`\ell_1` regularization
======================================================

In this example, we use CVXPY to train a logistic regression classifier
with :math:`\ell_1` regularization. We are given data :math:`(x_i,y_i)`,
:math:`i=1,\ldots, m`. The :math:`x_i \in {\bf R}^n` are feature
vectors, while the :math:`y_i \in \{0, 1\}` are associated boolean
classes.

Our goal is to construct a linear classifier
:math:`\hat y = \mathbb{1}[\beta^T x > 0]`, which is :math:`1` when
:math:`\beta^T x` is positive and :math:`0` otherwise. We model the
posterior probabilities of the classes given the data linearly, with

.. math::


   \log \frac{\mathrm{Pr} (Y=1 \mid X = x)}{\mathrm{Pr} (Y=0 \mid X = x)} = \beta^T x.

This implies that

.. math::


   \mathrm{Pr} (Y=1 \mid X = x) = \frac{\exp(\beta^T x)}{1 + \exp(\beta^T x)}, \quad
   \mathrm{Pr} (Y=0 \mid X = x) = \frac{1}{1 + \exp(\beta^T x)}.

We fit :math:`\beta` by maximizing the log-likelihood of the data, plus
a regularization term :math:`\lambda \|\beta\|_1` with
:math:`\lambda > 0`:

.. math::


   \ell(\beta) = \sum_{i=1}^{m} y_i \beta^T x_i - \log(1 + \exp (\beta^T x_i)) - \lambda \|\beta\|_1.

Because :math:`\ell` is a concave function of :math:`\beta`, this is a
convex optimization problem.

.. code:: python

    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt

In the following code we generate data with :math:`n=50` features by
randomly choosing :math:`x_i` and supplying a sparse
:math:`\beta_{\mathrm{true}} \in {\bf R}^n`. We then set
:math:`y_i = \mathbb{1}[\beta_{\mathrm{true}}^T x_i + z_i > 0]`, where
the :math:`z_i` are i.i.d. normal random variables. We divide the data
into training and test sets with :math:`m=50` examples each.

.. code:: python

    np.random.seed(1)
    n = 50
    m = 50
    def sigmoid(z):
      return 1/(1 + np.exp(-z))
    
    beta_true = np.array([1, 0.5, -0.5] + [0]*(n - 3))
    X = (np.random.random((m, n)) - 0.5)*10
    Y = np.round(sigmoid(X @ beta_true + np.random.randn(m)*0.5))
    
    X_test = (np.random.random((2*m, n)) - 0.5)*10
    Y_test = np.round(sigmoid(X_test @ beta_true + np.random.randn(2*m)*0.5))

We next formulate the optimization problem using CVXPY.

.. code:: python

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    log_likelihood = cp.sum(
        cp.multiply(Y, X @ beta) - cp.logistic(X @ beta) 
    )
    problem = cp.Problem(cp.Maximize(log_likelihood/m - lambd * cp.norm(beta, 1)))

We solve the optimization problem for a range of :math:`\lambda` to
compute a trade-off curve. We then plot the train and test error over
the trade-off curve. A reasonable choice of :math:`\lambda` is the value
that minimizes the test error.

.. code:: python

    def error(scores, labels):
      scores[scores > 0] = 1
      scores[scores <= 0] = 0
      return np.sum(np.abs(scores - labels)) / float(np.size(labels))

.. code:: python

    trials = 100
    train_error = np.zeros(trials)
    test_error = np.zeros(trials)
    lambda_vals = np.logspace(-2, 0, trials)
    beta_vals = []
    for i in range(trials):
        lambd.value = lambda_vals[i]
        problem.solve()
        train_error[i] = error( (X @ beta).value, Y)
        test_error[i] = error( (X_test @ beta).value, Y_test)
        beta_vals.append(beta.value)

.. code:: python

    %matplotlib inline
    %config InlineBackend.figure_format = "svg"
    
    plt.plot(lambda_vals, train_error, label="Train error")
    plt.plot(lambda_vals, test_error, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.show()



.. image:: logistic_regression_files/logistic_regression_9_0.svg


We also plot the regularization path, or the :math:`\beta_i` versus
:math:`\lambda`. Notice that a few features remain non-zero longer for
larger :math:`\lambda` than the rest, which suggests that these features
are the most important.

.. code:: python

    for i in range(n):
        plt.plot(lambda_vals, [wi for wi in beta_vals])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")



.. image:: logistic_regression_files/logistic_regression_11_0.svg


We plot the true :math:`\beta` versus reconstructed :math:`\beta`, as
chosen to minimize error on the test set. The non-zero coefficients are
reconstructed with good accuracy. There are a few values in the
reconstructed :math:`\beta` that are non-zero but should be zero.

.. code:: python

    idx = np.argmin(test_error)
    plt.plot(beta_true, label=r"True $\beta$")
    plt.plot(beta_vals[idx], label=r"Reconstructed $\beta$")
    plt.xlabel(r"$i$", fontsize=16)
    plt.ylabel(r"$\beta_i$", fontsize=16)
    plt.legend(loc="upper right")




.. parsed-literal::

    <matplotlib.legend.Legend at 0x108adedd8>




.. image:: logistic_regression_files/logistic_regression_13_1.svg

