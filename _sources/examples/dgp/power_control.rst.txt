
Power control
=============

*This example is adapted from Boyd, Kim, Vandenberghe, and Hassibi,* "`A
Tutorial on Geometric
Programming <https://web.stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`__."

*The problem data is adapted from the corresponding example in CVX's
example library (Almir Mutapcic).*

This example formulates and solves a power control problem for
communication systems, in which the goal is to minimize the total
transmitter power across n trasmitters, each trasmitting positive power
levels :math:`P_1`, :math:`P_2`, :math:`\ldots`, :math:`P_n` to
:math:`n` receivers, labeled :math:`1, \ldots, n`, with receiver
:math:`i` receiving signal from transmitter :math:`i`.

The power received from transmitter :math:`j` at receiver :math:`i` is
:math:`G_{ij} P_{j}`, where :math:`G_{ij} > 0` represents the path gain
from transmitter :math:`j` to receiver :math:`i`. The signal power at
receiver :math:`i` is :math:`G_{ii} P_i`, and the interference power at
receiver :math:`i` is :math:`\sum_{k \neq i} G_{ik}P_k`. The noise power
at receiver :math:`i` is :math:`\sigma_i`, and the signal to noise ratio
(SINR) of the :math:`i`\ th receiver-transmitter pair is

.. math::


   S_i = \frac{G_{ii}P_i}{\sigma_i + \sum_{k \neq i} G_{ik}P_k}.

The transmitters and receivers are constrained to have a minimum SINR
:math:`S^{\text{min}}`, and the :math:`P_i` are bounded between
:math:`P_i^{\text{min}}` and :math:`P_i^{\text{max}}`. This gives the
problem

.. math::


   \begin{array}{ll}
   \mbox{minimize} & P_1 + \cdots + P_n \\
   \mbox{subject to} & P_i^{\text{min}} \leq P_i \leq P_i^{\text{max}}, \\
   & 1/S^{\text{min}} \geq \frac{\sigma_i + \sum_{k \neq i} G_{ik}P_k}{G_{ii}P_i}.
   \end{array}

.. code:: python

    import cvxpy as cp
    import numpy as np
    
    # Problem data
    n = 5                     # number of transmitters and receivers
    sigma = 0.5 * np.ones(n)  # noise power at the receiver i
    p_min = 0.1 * np.ones(n)  # minimum power at the transmitter i
    p_max = 5 * np.ones(n)    # maximum power at the transmitter i
    sinr_min = 0.2            # threshold SINR for each receiver
    
    # Path gain matrix
    G = np.array(
       [[1.0, 0.1, 0.2, 0.1, 0.05],
        [0.1, 1.0, 0.1, 0.1, 0.05],
        [0.2, 0.1, 1.0, 0.2, 0.2],
        [0.1, 0.1, 0.2, 1.0, 0.1],
        [0.05, 0.05, 0.2, 0.1, 1.0]])
    p = cp.Variable(shape=(n,), pos=True)
    objective = cp.Minimize(cp.sum(p))
    
    S_p = []
    for i in range(n):
        S_p.append(cp.sum(cp.hstack(G[i, k]*p[k] for k in range(n) if i != k)))
    S = sigma + cp.hstack(S_p)
    signal_power = cp.multiply(cp.diag(G), p)
    inverse_sinr = S/signal_power
    constraints = [
        p >= p_min, 
        p <= p_max,
        inverse_sinr <= (1/sinr_min),
    ]
    
    problem = cp.Problem(objective, constraints)

.. code:: python

    problem.is_dgp()




.. parsed-literal::

    True



.. code:: python

    problem.solve(gp=True)
    problem.value




.. parsed-literal::

    0.9615384629119621



.. code:: python

    p.value




.. parsed-literal::

    array([0.18653846, 0.16730769, 0.23461538, 0.19615385, 0.17692308])



.. code:: python

    inverse_sinr.value




.. parsed-literal::

    array([5., 5., 5., 5., 5.])



.. code:: python

    (1/sinr_min)




.. parsed-literal::

    5.0


