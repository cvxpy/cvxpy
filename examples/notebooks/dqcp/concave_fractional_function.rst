
Fractional optimization
=======================

This notebook shows how to solve a simple *concave fractional problem*,
in which the objective is to maximize the ratio of a nonnegative concave
function and a positive convex function. Concave fractional problems are
quasiconvex programs (QCPs). They can be specified using disciplined
quasiconvex programming
(`DQCP <https://www.cvxpy.org/tutorial/dqcp/index.html>`__), and hence
can be solved using CVXPY.

.. code:: 

    !pip install --upgrade cvxpy


.. parsed-literal::

    Requirement already up-to-date: cvxpy in /usr/local/lib/python3.6/dist-packages (1.0.23)
    Requirement already satisfied, skipping upgrade: scs>=1.1.3 in /usr/local/lib/python3.6/dist-packages (from cvxpy) (2.1.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /usr/local/lib/python3.6/dist-packages (from cvxpy) (0.70.7)
    Requirement already satisfied, skipping upgrade: numpy>=1.15 in /usr/local/lib/python3.6/dist-packages (from cvxpy) (1.16.3)
    Requirement already satisfied, skipping upgrade: scipy>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from cvxpy) (1.3.0)
    Requirement already satisfied, skipping upgrade: ecos>=2 in /usr/local/lib/python3.6/dist-packages (from cvxpy) (2.0.7.post1)
    Requirement already satisfied, skipping upgrade: osqp>=0.4.1 in /usr/local/lib/python3.6/dist-packages (from cvxpy) (0.5.0)
    Requirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.6/dist-packages (from cvxpy) (1.12.0)
    Requirement already satisfied, skipping upgrade: dill>=0.2.9 in /usr/local/lib/python3.6/dist-packages (from multiprocess->cvxpy) (0.2.9)
    Requirement already satisfied, skipping upgrade: future in /usr/local/lib/python3.6/dist-packages (from osqp>=0.4.1->cvxpy) (0.16.0)


.. code:: 

    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt

Our goal is to minimize the function

.. math:: \frac{\sqrt{x}}{\exp(x)}.

This function is not concave, but it is quasiconcave, as can be seen by
inspecting its graph.

.. code:: 

    plt.plot([np.sqrt(y) / np.exp(y) for y in np.linspace(0, 10)])
    plt.show()



.. image:: concave_fractional_function_files/concave_fractional_function_4_0.png


The below code specifies and solves the QCP, using DQCP. The concave
fraction function is DQCP-compliant, because the ratio atom is
quasiconcave (actually, quasilinear), increasing in the numerator when
the denominator is positive, and decreasing in the denominator when the
numerator is nonnegative.

.. code:: 

    x = cp.Variable()
    concave_fractional_fn = cp.sqrt(x) / cp.exp(x)
    problem = cp.Problem(cp.Maximize(concave_fractional_fn))
    assert problem.is_dqcp()
    problem.solve(qcp=True)




.. parsed-literal::

    0.4288821220397949



.. code:: 

    x.value




.. parsed-literal::

    array(0.50000165)


