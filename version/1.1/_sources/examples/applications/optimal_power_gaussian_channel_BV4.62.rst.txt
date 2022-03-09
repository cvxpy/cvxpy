
Optimal Power and Bandwidth Allocation in a Gaussian Channel
============================================================

by Robert Gowers, Roger Hill, Sami Al-Izzi, Timothy Pollington and Keith
Briggs

from Boyd and Vandenberghe, Convex Optimization, exercise 4.62 page 210

Consider a system in which a central node transmits messages to
:math:`n` receivers. Each receiver channel :math:`i \in \{1,...,n\}` has
a transmit power :math:`P_i` and bandwidth :math:`W_i`. A fraction of
the total power and bandwidth is allocated to each channel, such that
:math:`\sum_{i=1}^{n}P_i = P_{tot}` and
:math:`\sum_{i=1}^{n}W_i = W_{tot}`. Given some utility function of the
bit rate of each channel, :math:`u_i(R_i)`, the objective is to maximise
the total utility :math:`U = \sum_{i=1}^{n}u_i(R_i)`.

Assuming that each channel is corrupted by Gaussian white noise, the
signal to noise ratio is given by :math:`\beta_i P_i/W_i`. This means
that the bit rate is given by:

:math:`R_i = \alpha_i W_i \log_2(1+\beta_iP_i/W_i)`

where :math:`\alpha_i` and :math:`\beta_i` are known positive constants.

One of the simplest utility functions is the data rate itself, which
also gives a convex objective function.

The optimisation problem can be thus be formulated as:

minimise :math:`\sum_{i=1}^{n}-\alpha_i W_i \log_2(1+\beta_iP_i/W_i)`

subject to
:math:`\sum_{i=1}^{n}P_i = P_{tot} \quad \sum_{i=1}^{n}W_i = W_{tot} \quad P \succeq 0 \quad W \succeq 0`

Although this is a convex optimisation problem, it must be rewritten in
DCP form since :math:`P_i` and :math:`W_i` are variables and DCP
prohibits dividing one variable by another directly. In order to rewrite
the problem in DCP format, we utilise the :math:`\texttt{kl_div}`
function in CVXPY, which calculates the Kullback-Leibler divergence.

:math:`\text{kl_div}(x,y) = x\log(x/y)-x+y`

:math:`-R_i = \text{kl_div}(\alpha_i W_i, \alpha_i(W_i+\beta_iP_i)) - \alpha_i\beta_iP_i`

Now that the objective function is in DCP form, the problem can be
solved using CVXPY.

.. code:: python

    #!/usr/bin/env python3
    # @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
    
    import numpy as np
    import cvxpy as cp

.. code:: python

    def optimal_power(n, a_val, b_val, P_tot=1.0, W_tot=1.0):
        # Input parameters: α and β are constants from R_i equation
        n = len(a_val)
        if n != len(b_val):
            print('alpha and beta vectors must have same length!')
            return 'failed', np.nan, np.nan, np.nan
        
        P = cp.Variable(shape=n)
        W = cp.Variable(shape=n)
        alpha = cp.Parameter(shape=n)
        beta = cp.Parameter(shape=n)
        alpha.value = np.array(a_val)
        beta.value = np.array(b_val)
    
        # This function will be used as the objective so must be DCP; 
        # i.e. elementwise multiplication must occur inside kl_div, 
        # not outside otherwise the solver does not know if it is DCP...
        R = cp.kl_div(cp.multiply(alpha, W),
                      cp.multiply(alpha, W + cp.multiply(beta, P))) - \
                      cp.multiply(alpha, cp.multiply(beta, P))
    
        objective = cp.Minimize(cp.sum(R))
        constraints = [P>=0.0,
                       W>=0.0,
                       cp.sum(P)-P_tot==0.0,
                       cp.sum(W)-W_tot==0.0]
        
        prob = cp.Problem(objective, constraints)
        prob.solve()
          
        return prob.status, -prob.value, P.value, W.value

Example
-------

Consider the case where there are 5 channels, :math:`n=5`,
:math:`\alpha = \beta = (2.0,2.2,2.4,2.6,2.8)`,
:math:`P_{\text{tot}} = 0.5` and :math:`W_{\text{tot}}=1`.

.. code:: python

    np.set_printoptions(precision=3)
    n = 5               # number of receivers in the system
    
    a_val = np.arange(10,n+10)/(1.0*n)  # α
    b_val = np.arange(10,n+10)/(1.0*n)  # β
    P_tot = 0.5
    W_tot = 1.0
    status, utility, power, bandwidth = optimal_power(n, a_val, b_val, P_tot, W_tot)
    
    print('Status: {}'.format(status))
    print('Optimal utility value = {:.4g}'.format(utility))
    print('Optimal power level:\n{}'.format(power))
    print('Optimal bandwidth:\n{}'.format(bandwidth))


.. parsed-literal::

    Status: optimal
    Optimal utility value = 2.451
    Optimal power level:
    [1.151e-09 1.708e-09 2.756e-09 5.788e-09 5.000e-01]
    Optimal bandwidth:
    [3.091e-09 3.955e-09 5.908e-09 1.193e-08 1.000e+00]

