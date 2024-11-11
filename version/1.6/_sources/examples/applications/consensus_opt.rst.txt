
Consensus optimization
======================

Suppose we have a convex optimization problem with :math:`N` terms in
the objective

.. math::

   \begin{array}{ll} \mbox{minimize} & \sum_{i=1}^N f_i(x)\\
   \end{array}

For example, we might be fitting a model to data and :math:`f_i` is the
loss function for the :math:`i`\ th block of training data.

We can convert this problem into consensus form

.. math::

   \begin{array}{ll} \mbox{minimize} & \sum_{i=1}^N f_i(x_i)\\
   \mbox{subject to} & x_i = z
   \end{array}

We interpret the :math:`x_i` as local variables, since they are
particular to a given :math:`f_i`. The variable :math:`z`, by contrast,
is global. The constraints :math:`x_i = z` enforce consistency, or
consensus.

We can solve a problem in consensus form using the Alternating Direction
Method of Multipliers (ADMM). Each iteration of ADMM reduces to the
following updates:

.. math::

   \begin{array}{lll}
   % xbar, u parameters in prox.
   % called proximal operator.
   x^{k+1}_i & := & \mathop{\rm argmin}_{x_i}\left(f_i(x_i) + (\rho/2)\left\|x_i - \overline{x}^k + u^k_i \right\|^2_2 \right) \\
   % u running sum of errors.
   u^{k+1}_i & := & u^{k}_i + x^{k+1}_i - \overline{x}^{k+1}
   \end{array}

where :math:`\overline{x}^k = (1/N)\sum_{i=1}^N x^k_i`.

The following code carries out consensus ADMM, using CVXPY to solve the
local subproblems.

We split the :math:`x_i` variables across :math:`N` different worker
processes. The workers update the :math:`x_i` in parallel. A master
process then gathers and averages the :math:`x_i` and broadcasts
:math:`\overline x` back to the workers. The workers update :math:`u_i`
locally.

.. code:: 

    from cvxpy import *
    import numpy as np
    from multiprocessing import Process, Pipe
    
    # Number of terms f_i.
    N = ...
    # A list of all the f_i.
    f_list = ...
    
    def run_worker(f, pipe):
        xbar = Parameter(n, value=np.zeros(n))
        u = Parameter(n, value=np.zeros(n))
        f += (rho/2)*sum_squares(x - xbar + u)
        prox = Problem(Minimize(f))
        # ADMM loop.
        while True:
            prox.solve()
            pipe.send(x.value)
            xbar.value = pipe.recv()
            u.value += x.value - xbar.value
    
    # Setup the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_process, args=(f_list[i], remote))]
        procs[-1].start()
    
    # ADMM loop.
    for i in range(MAX_ITER):
        # Gather and average xi
        xbar = sum(pipe.recv() for pipe in pipes)/N
        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)
    
    [p.terminate() for p in procs]
