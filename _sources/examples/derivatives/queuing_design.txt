
Queuing design
==============

In this example, we consider the design of an :math:`M/M/N` queuing
system, and we study the sensitivity of the design with respect to
various parameters, such as the maximum total delay and total service
rate.

This example was described in the paper `Differentiating through Log-Log
Convex
Programs <http://web.stanford.edu/~boyd/papers/pdf/diff_llcvx.pdf>`__.

We consider the optimization of a (Markovian) queuing system, with
:math:`N` queues. A queuing system is a collection of queues, in which
queued items wait to be served; the queued items might be threads in an
operating system, or packets in an input or output buffer of a
networking system. A natural goal to minimize the service load of the
system, given constraints on various properties of the queuing system,
such as limits on the maximum delay or latency. In this example, we
formulate this design problem as a log-log convex program (LLCP), and
compute the sensitivity of the design variables with respect to the
parameters. The queuing system under consideration here is known as an
:math:`M/M/N` queue.

We assume that items arriving at the :math:`i`\ th queue are generated
by a Poisson process with rate :math:`\lambda_i`, and that the service
times for the :math:`i`\ th queue follow an exponential distribution
with parameter :math:`\mu_i`, for :math:`i=1, \ldots, N`. The of the
queuing system is a function
:math:`\ell : \mathbf{R}^{N}_{++} \times \mathbf{R}^{N}_{++} \to \mathbf{R}^{N}_{++}`
of the arrival rate vector :math:`\lambda` and the service rate vector
:math:`\mu`, with components

.. math::


   \ell_i(\lambda, \mu) = \frac{\mu_i}{\lambda_i}, \quad i=1, \ldots, N.

(This is the reciprocal of the traffic load, which is usually denoted by
:math:`\rho`.) Similarly, the queue occupancy, the average delay, and
the total delay of the system are (respectively) functions :math:`q`,
:math:`w`, and :math:`d` of :math:`\lambda` and :math:`\mu`, with
components

.. math::


   q_i(\lambda, \mu) =
   \frac{\ell_i(\lambda, \mu)^{-2}}{1 - \ell_i(\lambda, \mu)^{-1}}, \quad
   w_i(\lambda, \mu) = \frac{q_i(\lambda, \mu)}{\lambda_i} + \frac{1}{\mu_i}, \quad
   d_i(\lambda, \mu) = \frac{1}{\mu_i - \lambda_i}

These functions have domain
:math:`\{(\lambda, \mu) \in \mathbf{R}^{N}_{++} \times \mathbf{R}^{N}_{++} \mid \lambda < \mu \}`,
where the inequality is meant elementwise. The queuing system has limits
on the queue occupancy, average queuing delay, and total delay, which
must satisfy

.. math::


   q(\lambda, \mu) \leq q_{\max}, \quad w(\lambda, \mu) \leq w_{\max}, \quad d(\lambda, \mu) \leq d_{\max},

where :math:`q_{\max}`, :math:`w_{\max}`, and
:math:`d_{\max} \in \mathbf{R}^{N}_{++}` are parameters and the
inequalities are meant elementwise. Additionally, the arrival rate
vector :math:`\lambda` must be at least
:math:`\lambda_{\mathrm{min}} \in \mathbf{R}^{N}_{++}`, and the sum of
the service rates must be no greater than
:math:`\mu_{\max} \in \mathbf{R}_{++}`.

Our design problem is to choose the arrival rates and service times to
minimize a weighted sum of the service loads,
:math:`\gamma^T \ell(\lambda, \mu)`, where
:math:`\gamma \in \mathbf{R}^{N}_{++}` is the weight vector, while
satisfying the constraints. The problem is

.. math::


   \begin{array}{ll}
   \mbox{minimize} & \gamma^T \ell(\lambda, \mu) \\
   \mbox{subject to}
   & q(\lambda, \mu) \leq q_{\max} \\
   & w(\lambda, \mu) \leq w_{\max} \\
   & d(\lambda, \mu) \leq d_{\max} \\
   & \lambda \geq \lambda_{\mathrm{min}}, \quad
   \sum_{i=1}^{N} \mu_i \leq \mu_{\max}.
   \end{array}

Here, :math:`\lambda, \mu \in \mathbf{R}^{N}_{++}` are the variables and
:math:`\gamma, q_{\max}, w_{\max}, d_{\max}, \lambda_{\mathrm{min}} \in \mathbf{R}^{N}_{++}`
and :math:`\mu_{\max} \in \mathbf{R}_{++}` are the parameters. This
problem is an LLCP. The objective function is a posynomial, as is the
constraint function :math:`w`. The functions :math:`d` and :math:`q` are
not posynomials, but they are log-log convex; log-log convexity of
:math:`d` follows from the composition rule, since the function
:math:`(x, y) \mapsto y - x` is log-log concave (for :math:`0 < x < y`),
and the ratio :math:`(x, y) \mapsto x/y` is log-log affine and
decreasing in :math:`y`. By a similar argument, :math:`q` is also
log-log convex.

.. code:: ipython3

    import cvxpy as cp
    import numpy as np
    import time
    
    
    mu = cp.Variable(pos=True, shape=(2,), name='mu')
    lam = cp.Variable(pos=True, shape=(2,), name='lambda')
    ell = cp.Variable(pos=True, shape=(2,), name='ell')
    
    w_max = cp.Parameter(pos=True, shape=(2,), value=np.array([2.5, 3.0]), name='w_max')
    d_max = cp.Parameter(pos=True, shape=(2,), value=np.array([2., 2.]), name='d_max')
    q_max = cp.Parameter(pos=True, shape=(2,), value=np.array([4., 5.0]), name='q_max')
    lam_min = cp.Parameter(pos=True, shape=(2,), value=np.array([0.5, 0.8]), name='lambda_min')
    mu_max = cp.Parameter(pos=True, value=3.0, name='mu_max')
    gamma = cp.Parameter(pos=True, shape=(2,), value=np.array([1.0, 2.0]), name='gamma')
                         
    lq = (ell)**(-2)/cp.one_minus_pos(ell**(-1))
    q = lq
    w = lq/lam + 1/mu
    d = 1/cp.diff_pos(mu, lam)                    
    
    constraints = [
        w <= w_max,
        d <= d_max,
        q <= q_max,
        lam >= lam_min,
        cp.sum(mu) <= mu_max,
        ell == mu/lam,
    ]
    
    objective_fn = gamma.T @ ell
    
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(requires_grad=True, gp=True, eps=1e-14, max_iters=10000, mode='dense')




.. parsed-literal::

    4.457106781186705



The solution is printed below.

.. code:: ipython3

    print('mu ', mu.value)
    print('lam ', lam.value)
    print('ell ', ell.value)


.. parsed-literal::

    mu  [1.32842713 1.67157287]
    lam  [0.82842712 1.17157287]
    ell  [1.60355339 1.4267767 ]


Sensitvities
~~~~~~~~~~~~

We compute the derivative of each variable with respect the parameters.
One takeaway is that the solution is highly sensitive to the values of
:math:`d_{\max}` and :math:`\mu_{\max}`, and that increasing these
parameters would decrease the service loads, especially on the first
queue.

.. code:: ipython3

    problem.solve(requires_grad=True, gp=True, eps=1e-14, max_iters=10000, mode='dense')
    
    for var in [lam, mu, ell]:
        print('Variable ', var.name())
        print('Gradient with respect to first component')
        var.gradient = np.array([1., 0.])
        problem.backward()
        for param in problem.parameters():
            if np.prod(param.shape) == 2:
                print('{0}: {1:.3g}, {2:.3g}'.format(param.name(), param.gradient[0], param.gradient[1]))
            else:
                print('{0}: {1:.3g}'.format(param.name(), param.gradient))
    
        print('Gradient with respect to second component')
        var.gradient = np.array([0., 1.])
        problem.backward()
        for param in problem.parameters():
            if np.prod(param.shape) == 2:
                print('{0}: {1:.3g}, {2:.3g}'.format(param.name(), param.gradient[0], param.gradient[1]))
            else:
                print('{0}: {1:.3g}'.format(param.name(), param.gradient))
        
        var.gradient = np.zeros(2)
        print('')


.. parsed-literal::

    Variable  lambda
    Gradient with respect to first component
    gamma: 0.213, -0.107
    w_max: 5.43e-12, 5.64e-12
    d_max: -0.411, -0.113
    q_max: 5.99e-12, 4.77e-12
    lambda_min: -1.56e-11, -7.35e-12
    mu_max: 0.927
    Gradient with respect to second component
    gamma: -0.458, 0.229
    w_max: 2.08e-11, 2.16e-11
    d_max: -0.105, -0.466
    q_max: 2.29e-11, 1.83e-11
    lambda_min: -5.97e-11, -2.82e-11
    mu_max: 1.01
    
    Variable  mu
    Gradient with respect to first component
    gamma: 0.213, -0.107
    w_max: 1.55e-11, 1.6e-11
    d_max: -0.661, -0.113
    q_max: 1.7e-11, 1.36e-11
    lambda_min: -4.43e-11, -2.09e-11
    mu_max: -0.0727
    Gradient with respect to second component
    gamma: -0.458, 0.229
    w_max: 2.3e-11, 2.39e-11
    d_max: -0.105, -0.716
    q_max: 2.53e-11, 2.02e-11
    lambda_min: -6.59e-11, -3.11e-11
    mu_max: 0.00996
    
    Variable  ell
    Gradient with respect to first component
    gamma: -0.245, 0.122
    w_max: 2e-11, 2.08e-11
    d_max: -0.282, -0.22
    q_max: 2.21e-11, 1.76e-11
    lambda_min: -5.74e-11, -2.71e-11
    mu_max: -0.334
    Gradient with respect to second component
    gamma: 0.122, -0.0611
    w_max: -1.24e-13, -1.29e-13
    d_max: -0.101, -0.195
    q_max: -1.37e-13, -1.09e-13
    lambda_min: 3.58e-13, 1.66e-13
    mu_max: -0.197
    


Perturbation analysis
---------------------

Next, we perturb each parameter by a small amount, and compare the
prediction of a first-order approximation to the solution of the
perturbed problem to the true solution.

.. code:: ipython3

    problem.solve(requires_grad=True, gp=True, eps=1e-14, max_iters=10000, mode='dense')
    
    mu_value = mu.value
    lam_value = lam.value
    
    delta = 0.01
    for param in problem.parameters():
        param.delta = param.value * delta
        
    problem.derivative()
    
    lam_pred = (lam.delta / lam_value) * 100
    mu_pred = (mu.delta / mu_value) * 100
    
    print('lam predicted (percent change): ', lam_pred)
    print('mu predicted (percent change): ', mu_pred)
    
    
    for param in problem.parameters():
        param._old_value = param.value
        param.value += param.delta
    problem.solve(cp.SCS, gp=True, eps=1e-14, max_iters=10000)
    
    lam_actual = ((lam.value - lam_value) / lam_value) * 100
    mu_actual = ((mu.value - mu_value) / mu_value) * 100
        
    print('lam actual (percent change): ', lam_actual)
    print('mu actual (percent change): ', mu_actual)


.. parsed-literal::

    lam predicted (percent change):  [2.32203282 1.77228841]
    mu predicted (percent change):  [1.07166961 0.94304296]
    lam actual (percent change):  [1.99504983 1.99504965]
    mu actual (percent change):  [0.87148458 1.10213353]

