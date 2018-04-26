
Fault detection
===============

We'll consider a problem of identifying faults that have occurred in a
system based on sensor measurements of system performance.

Topic references
================

-  `Samar, Sikandar, Dimitry Gorinevsky, and Stephen Boyd. "Likelihood
   Bounds for Constrained Estimation with Uncertainty." Decision and
   Control, 2005 and 2005 European Control Conference. CDC-ECC'05. 44th
   IEEE Conference on. IEEE,
   2005. <http://web.stanford.edu/~boyd/papers/pdf/map_bounds.pdf>`__

Problem statement
=================

Each of :math:`n` possible faults occurs independently with probability
:math:`p`. The vector :math:`x \in \lbrace 0,1 \rbrace^{n}` encodes the
fault occurrences, with :math:`x_i = 1` indicating that fault :math:`i`
has occurred. System performance is measured by :math:`m` sensors. The
sensor output is

.. raw:: latex

   \begin{equation}
   y = Ax + v = \sum_{i=1}^n a_i x_i + v,
   \end{equation}

where :math:`A \in \mathbf{R}^{m \times n}` is the sensing matrix with
column :math:`a_i` being the **fault signature** of fault :math:`i`, and
:math:`v \in \mathbf{R}^m` is a noise vector where :math:`v_j` is
Gaussian with mean 0 and variance :math:`\sigma^2`.

The objective is to guess :math:`x` (which faults have occurred) given
:math:`y` (sensor measurements).

We are interested in the setting where :math:`n > m`, that is, when we
have more possible faults than measurements. In this setting, we can
expect a good recovery when the vector :math:`x` is sparse. This is the
subject of compressed sensing.

Solution approach
=================

To identify the faults, one reasonable approach is to choose
:math:`x \in \lbrace 0,1 \rbrace^{n}` to minimize the negative
log-likelihood function

.. raw:: latex

   \begin{equation}
   \ell(x) = \frac{1}{2 \sigma^2} \|Ax-y\|_2^2 +  \log(1/p-1)\mathbf{1}^T x + c.
   \end{equation}

However, this problem is nonconvex and NP-hard, due to the constraint
that :math:`x` must be Boolean.

To make this problem tractable, we can relax the Boolean constraints and
instead constrain :math:`x_i \in [0,1]`.

The optimization problem

.. raw:: latex

   \begin{array}{ll}
   \mbox{minimize} &  \|Ax-y\|_2^2 + 2 \sigma^2 \log(1/p-1)\mathbf{1}^T x\\
   \mbox{subject to} &  0 \leq x_i \leq 1, \quad i=1, \ldots n
   \end{array}

is convex. We'll refer to the solution of the convex problem as the
**relaxed ML** estimate.

By taking the relaxed ML estimate of :math:`x` and rounding the entries
to the nearest of 0 or 1, we recover a Boolean estimate of the fault
occurrences.

Example
=======

We'll generate an example with :math:`n = 2000` possible faults,
:math:`m = 200` measurements, and fault probability :math:`p = 0.01`.
We'll choose :math:`\sigma^2` so that the signal-to-noise ratio is 5.
That is,

.. raw:: latex

   \begin{equation}
   \sqrt{\frac{\mathbf{E}\|Ax \|^2_2}{\mathbf{E} \|v\|_2^2}} = 5.
   \end{equation}

.. code:: 

    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(1)
    
    n = 2000
    m = 200
    p = 0.01
    snr = 5
    
    sigma = np.sqrt(p*n/(snr**2))
    A = np.random.randn(m,n)
    
    x_true = (np.random.rand(n) <= p).astype(np.int)
    v = sigma*np.random.randn(m)
    
    y = A.dot(x_true) + v

Below, we show :math:`x`, :math:`Ax` and the noise :math:`v`.

.. code:: 

    plt.plot(range(n),x_true)




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x110695dd0>]




.. image:: fault_detection_files/fault_detection_3_1.png


.. code:: 

    plt.plot(range(m), A.dot(x_true),range(m),v)
    plt.legend(('Ax','v'))




.. parsed-literal::

    <matplotlib.legend.Legend at 0x110ee63d0>




.. image:: fault_detection_files/fault_detection_4_1.png


Recovery
========

We solve the relaxed maximum likelihood problem with CVXPY and then
round the result to get a Boolean solution.

.. code:: 

    %%time
    from cvxpy import *
    x = Variable(shape=(n,1))
    tau = 2*log(1/p - 1)*sigma**2
    obj = Minimize(sum_squares(A*x - y) + tau*sum(x))
    const = [0 <= x, x <= 1]
    Problem(obj,const).solve(verbose=True)
    
    # relaxed ML estimate
    x_rml = np.array(x.value).flatten()
    
    # rounded solution
    x_rnd = (x_rml >= .5).astype(int)


.. parsed-literal::

    
    ECOS 1.0.4 - (c) A. Domahidi, Automatic Control Laboratory, ETH Zurich, 2012-2014.
    
    It     pcost         dcost      gap     pres    dres     k/t     mu      step     IR
     0   +7.127e+03   -6.144e+04   +8e+05   8e+00   1e-01   1e+00   2e+02    N/A     1 1 -
     1   +7.014e+02   -1.137e+04   +4e+05   1e+00   2e-02   4e+01   1e+02   0.9899   1 1 1
     2   +5.300e+01   -1.510e+03   +9e+04   2e-01   2e-03   2e+01   2e+01   0.9406   2 1 1
     3   +1.140e+02   -7.533e+02   +5e+04   1e-01   1e-03   1e+01   1e+01   0.5426   2 2 2
     4   +1.378e+02   -3.905e+02   +3e+04   6e-02   8e-04   5e+00   8e+00   0.5017   2 2 2
     5   +1.391e+02   -2.656e+02   +3e+04   5e-02   6e-04   3e+00   7e+00   0.4344   2 2 1
     6   +1.645e+02   +8.938e+00   +1e+04   2e-02   2e-04   9e-01   3e+00   0.6950   2 2 2
     7   +1.740e+02   +7.476e+01   +6e+03   1e-02   2e-04   5e-01   2e+00   0.5070   2 2 2
     8   +1.739e+02   +7.682e+01   +6e+03   1e-02   2e-04   4e-01   2e+00   0.0978   3 2 1
     9   +1.844e+02   +1.482e+02   +2e+03   4e-03   6e-05   2e-02   6e-01   0.9899   2 2 2
    10   +1.889e+02   +1.755e+02   +9e+02   2e-03   2e-05   9e-03   2e-01   0.7568   2 2 2
    11   +1.907e+02   +1.864e+02   +3e+02   5e-04   7e-06   3e-03   7e-02   0.8071   2 2 2
    12   +1.912e+02   +1.892e+02   +1e+02   2e-04   3e-06   1e-03   3e-02   0.8099   2 2 2
    13   +1.914e+02   +1.906e+02   +6e+01   1e-04   1e-06   5e-04   1e-02   0.7158   3 2 2
    14   +1.916e+02   +1.912e+02   +3e+01   4e-05   6e-07   2e-04   6e-03   0.8640   3 1 1
    15   +1.916e+02   +1.916e+02   +4e+00   7e-06   9e-08   3e-05   1e-03   0.8722   3 2 2
    16   +1.916e+02   +1.916e+02   +4e-01   7e-07   1e-08   4e-06   1e-04   0.9258   2 2 2
    17   +1.916e+02   +1.916e+02   +6e-02   1e-07   2e-09   5e-07   2e-05   0.8804   3 2 2
    18   +1.916e+02   +1.916e+02   +2e-02   4e-08   5e-10   2e-07   5e-06   0.7988   3 3 3
    19   +1.916e+02   +1.916e+02   +3e-03   6e-09   8e-11   3e-08   8e-07   0.9092   3 3 3
    20   +1.916e+02   +1.916e+02   +5e-04   9e-10   1e-11   5e-09   1e-07   0.9134   3 2 2
    21   +1.916e+02   +1.916e+02   +1e-04   2e-10   2e-12   9e-10   3e-08   0.8726   3 2 1
    22   +1.916e+02   +1.916e+02   +1e-05   2e-11   3e-13   1e-10   4e-09   0.9512   2 1 1
    
    OPTIMAL (within feastol=2.5e-11, reltol=7.3e-08, abstol=1.4e-05).
    Runtime: 4.225071 seconds.
    
    CPU times: user 4.66 s, sys: 123 ms, total: 4.78 s
    Wall time: 4.97 s


Evaluation
==========

We define a function for computing the estimation errors, and a function
for plotting :math:`x`, the relaxed ML estimate, and the rounded
solutions.

.. code:: 

    import matplotlib
    
    def errors(x_true, x, threshold=.5):
        '''Return estimation errors.
        
        Return the true number of faults, the number of false positives, and the number of false negatives.
        '''
        n = len(x_true)
        k = sum(x_true)
        false_pos = sum(np.logical_and(x_true < threshold, x >= threshold))
        false_neg = sum(np.logical_and(x_true >= threshold, x < threshold))
        return (k, false_pos, false_neg)
    
    def plotXs(x_true, x_rml, x_rnd, filename=None):
        '''Plot true, relaxed ML, and rounded solutions.'''
        matplotlib.rcParams.update({'font.size': 14})
        xs = [x_true, x_rml, x_rnd]
        titles = ['x_true', 'x_rml', 'x_rnd']
    
        n = len(x_true)
        k = sum(x_true)
    
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 3))
    
        for i,x in enumerate(xs):
                ax[i].plot(range(n), x)
                ax[i].set_title(titles[i])
                ax[i].set_ylim([0,1])
                
        if filename:
            fig.savefig(filename, bbox_inches='tight')
            
        return errors(x_true, x_rml,.5)

We see that out of 20 actual faults, the rounded solution gives perfect
recovery with 0 false negatives and 0 false positives.

.. code:: 

    plotXs(x_true, x_rml, x_rnd, 'fault.pdf')




.. parsed-literal::

    (20, 0, 0)




.. image:: fault_detection_files/fault_detection_10_1.png

