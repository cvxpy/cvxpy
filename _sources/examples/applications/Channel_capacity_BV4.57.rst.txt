Capacity of a Communication Channel
===================================

by Robert Gowers, Roger Hill, Sami Al-Izzi, Timothy Pollington and Keith
Briggs

from Boyd and Vandenberghe, Convex Optimization, exercise 4.57 pages
207-8

Convex optimization can be used to find the channel capacity :math:`C`
of a discrete memoryless channel. Consider a communication channel with
input :math:`X(t) \in \{1,2,...,n\}` and output
:math:`Y(t) \in \{1,2,...m\}`. This means that the random variables
:math:`X` and :math:`Y` can take :math:`n` and :math:`m` different
values, respectively.

In a discrete memoryless channel, the relation between the input and the
output is given by the transition probability:

:math:`p_{ij} = \mathbb{P}(Y(t)=i | X(t)=j)`

These transition probabilities form the channel transition matrix
:math:`P`, with :math:`P \in \mathbb{R}^{m\times n}`.

Assume that :math:`X` has a probability distribution denoted by
:math:`x \in \mathbb{R}^n`, meaning that:

:math:`x_j = \mathbb{P}(X(t) = j) \quad j \in \{1,...,n\}`.

From Shannon, the channel capacity is given by the maximum possible
mutual information :math:`I` between :math:`X` and :math:`Y`:

:math:`C = \sup_x I(X;Y)`

where,

:math:`I(X;Y) = -\sum_{i=1}^{m} y_i \log_2y_i + \sum_{j=1}^{n}\sum_{i=1}^{m}x_j p_{ij}\log_2p_{ij}`

Given that :math:`x\log x` is convex for :math:`x \geq 0`, we can
formulate this as a convex optimization problem:

minimise :math:`-I(X;Y)`

subject to :math:`\sum_{i=1}^{n}x_i = 1 \quad x \succeq 0 \quad` since
:math:`x` describes a probability

Due to the entropy function in CVXPY, this can be written quite easily
in DCP.

.. code:: ipython3

    #!/usr/bin/env python3
    # @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
    
    import cvxpy as cp
    import numpy as np
    import math

.. code:: ipython3

    def channel_capacity(n, m, P, sum_x=1):
        '''
        Boyd and Vandenberghe, Convex Optimization, exercise 4.57 page 207
        Capacity of a communication channel.
        
        We consider a communication channel, with input X(t)∈{1,..,n} and
        output Y(t)∈{1,...,m}, for t=1,2,... .The relation between the
        input and output is given statistically:
        p_(i,j) = ℙ(Y(t)=i|X(t)=j), i=1,..,m  j=1,...,n
        
        The matrix P ∈ ℝ^(m*n) is called the channel transition matrix, and
        the channel is called a discrete memoryless channel. Assuming X has a
        probability distribution denoted x ∈ ℝ^n, i.e.,
        x_j = ℙ(X=j), j=1,...,n
        
        The mutual information between X and Y is given by
        ∑(∑(x_j p_(i,j)log_2(p_(i,j)/∑(x_k p_(i,k)))))
        Then channel capacity C is given by
        C = sup I(X;Y).
        With a variable change of y = Px this becomes
        I(X;Y)=  c^T x - ∑(y_i log_2 y_i)
        where c_j = ∑(p_(i,j)log_2(p_(i,j)))
        '''
        
        # n is the number of different input values
        # m is the number of different output values
        if n*m == 0:
            print('The range of both input and output values must be greater than zero')
            return 'failed', np.nan, np.nan
    
        # x is probability distribution of the input signal X(t)
        x = cp.Variable(shape=n)
        
        # y is the probability distribution of the output signal Y(t)
        # P is the channel transition matrix
        y = P@x
        
        # I is the mutual information between x and y
        c = np.sum(np.array((xlogy(P, P) / math.log(2))), axis=0)
        I = c@x + cp.sum(cp.entr(y) / math.log(2))
    
        # Channel capacity maximised by maximising the mutual information
        obj = cp.Minimize(-I)
        constraints = [cp.sum(x) == sum_x,x >= 0]
        
        # Form and solve problem
        prob = cp.Problem(obj,constraints)
        prob.solve()
        if prob.status=='optimal':
            return prob.status, prob.value, x.value
        else:
            return prob.status, np.nan, np.nan
        

Example
-------

In this example we consider a communication channel with two possible
inputs and outputs, so :math:`n = m = 2`. The channel transition matrix
we use in this case is:

:math:`P = \pmatrix{0.75,0.25\\0.25,0.75}`

Note that the columns of :math:`P` must sum to 1 and all elements of
:math:`P` must be positive.

.. code:: ipython3

    np.set_printoptions(precision=3)
    n = 2
    m = 2
    P = np.array([[0.75,0.25],
                 [0.25,0.75]])
    stat, C, x = channel_capacity(n, m, P)
    print('Problem status: ',stat)
    print('Optimal value of C = {:.4g}'.format(C))
    print('Optimal variable x = \n', x)


.. parsed-literal::

    Problem status:  optimal
    Optimal value of C = 0.1181
    Optimal variable x = 
     [0.5 0.5]

