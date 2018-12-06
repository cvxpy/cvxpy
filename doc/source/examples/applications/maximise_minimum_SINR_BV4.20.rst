
Power Assignment in a Wireless Communication System
===================================================

by Robert Gowers, Roger Hill, Sami Al-Izzi, Timothy Pollington and Keith
Briggs

from Boyd and Vandenberghe, Convex Optimization, exercise 4.20 page 196

Convex optimization can be used to maximise the minimum signal to
inteference plus noise ratio (SINR) of a wireless communication system.
Consider a system with :math:`n` transmitters, each with power
:math:`p_j \geq 0`, transmitting to :math:`n` receivers. Let
:math:`G_{ij} \geq 0` denote the path gain from transmitter :math:`j` to
receiver :math:`i`. These path gains form the matrix
:math:`G \in \mathbb{R}^{n \times n}`.

Each receiver is assigned to a transmitter such that the signal power at
receiver :math:`i`, :math:`S_i = G_{ii}p_i` and the interefence power at
receiver :math:`i` is :math:`I_i = \sum_{k\neq i} G_{ik}p_k`. Given a
noise power :math:`\sigma_i` at each receiver, the SINR at receiver
:math:`i`, :math:`\gamma_i = \frac{S_i}{I_i + \sigma_i}`.

The objective is to maximise the minimum SINR of the system under
certain power constraints. These constraints are:

i - Each transmitter power :math:`p_j \leq P_j^{\text{max}}`

ii - If the transmitters are partitioned into :math:`m` nonoverlapping
groups, :math:`K_1, ..., K_m`, which share a common power supply with
total power :math:`P_l^{\text{gp}}`:
:math:`\sum_{k\in K_l}p_k \leq P_l^{\text{gp}}`.

iii - There is a maximum power that each receiver can receive
:math:`P_i^{\text{rc}}`,
:math:`\sum_{k=1}^{n}G_{ik}p_k \leq P_i^{\text{rc}}`.

The objective function can be rewritten as:

minimise :math:`\max_{i=1,...,n}\frac{I_i + \sigma_i}{S_i}`

However, since this is a quasiconvex objective function we cannot solve
it directly using CVXPY. Instead we must use a bisection method. First
we take the step of rewriting the objective,
:math:`\alpha = \gamma^{-1} \geq 0`, as a constraint:

:math:`I_i+\sigma_i \leq S_i\alpha`

Then we choose initial lower and upper bounds :math:`L_0` and
:math:`U_0` for :math:`\alpha`, which should be chosen such that
:math:`L < \alpha^* < U`, where :math:`\alpha^*` is the optimal value of
:math:`\alpha`. Starting with an initial value
:math:`\alpha_0 = \frac{1}{2}(L_0+U_0)`, feasibility is checked for
:math:`\alpha_0` by using an arbitrary objective function. The new upper
and lower bounds are determined from the feasibility:

If :math:`\alpha_0` is feasible then :math:`L_1 = L_0`,
:math:`U_1 = \alpha_0` and :math:`\alpha_1 = \frac{1}{2}(L_1+U_1)`.

If :math:`\alpha_0` is infeasible then :math:`L_1 = \alpha_1`,
:math:`U_1 = U_0` and :math:`\alpha_1 = \frac{1}{2}(L_1+U_1)`.

This bisection process is repeated until :math:`U_N - L_N < \epsilon`,
where :math:`\epsilon` is the desired tolerance.

.. code:: python

    #!/usr/bin/env python3
    # @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
    
    import cvxpy as cp
    import numpy as np

.. code:: python

    def maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max, epsilon = 0.001):
        # find n and m from the size of the path gain matrix
        n, m = np.shape(G)
        
        # Checks sizes of inputs
        if m != np.size(P_max):
            print('Error: P_max dimensions do not match gain matrix dimensions\n')
            return 'Error: P_max dimensions do not match gain matrix dimensions\n', np.nan, np.nan, np.nan
        
        if n != np.size(P_received):
            print('Error: P_received dimensions do not match gain matrix dimensions\n')
            return 'Error: P_received dimensions do not match gain matrix dimensions', np.nan, np.nan, np.nan
        
        if n != np.size(sigma):
            print('Error: σ dimensions do not match gain matrix dimensions\n')
            return 'Error: σ dimensions do not match gain matrix dimensions', np.nan, np.nan, np.nan
    
        #I = np.zeros((n,m))
        #S = np.zeros((n,m))
    
        delta = np.identity(n)
        S = G*delta # signal power matrix
        I = G-S # interference power matrix
    
        # group matrix: number of groups by number of transmitters
        num_groups = int(np.size(Group,0))
    
        if num_groups != np.size(Group_max):
            print('Error: Number of groups from Group matrix does not match dimensions of Group_max\n')
            return ('Error: Number of groups from Group matrix does not match dimensions of Group_max',
                    np.nan, np.nan, np.nan, np.nan)
    
        # normalising the max power of a group so it is in the range [0,1]
        Group_norm = Group/np.sum(Group,axis=1).reshape((num_groups,1))
        
        # create scalar optimisation variable p: the power of the n transmitters
        p = cp.Variable(shape=n)
        best = np.zeros(n)
    
        # set upper and lower bounds for sub-level set
        u = 1e4
        l = 0
    
        # alpha defines the sub-level sets of the generalised linear fractional problem
        # in this case α is the reciprocal of the minimum SINR
        alpha = cp.Parameter(shape=1)
        
        # set up the constraints for the bisection feasibility test
        constraints = [I*p + sigma <= alpha*S*p, p <= P_max, p >= 0, G*p <= P_received, Group_norm*p <= Group_max]
    
        # define objective function, in our case it's constant as only want to test the solution's feasibility
        obj = cp.Minimize(alpha)
        
        # now check whether the solution lies between u and l
        alpha.value = [u]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        if prob.status != 'optimal':
            # in this case the level set u is below the solution
            print('No optimal solution within bounds\n')
            return 'Error: no optimal solution within bounds', np.nan, np.nan, np.nan
        
        alpha.value = [l]
        prob = cp.Problem(obj, constraints)
        prob.solve()
    
        if prob.status == 'optimal':
            # in this case the level set l is below the solution
            print('No optimal solution within bounds\n')
            return 'Error: no optimal solution within bounds', np.nan, np.nan, np.nan
        
        # Bisection algortithm starts
        maxLoop = int(1e7)
        for i in range(1,maxLoop):
            # First check that u is in the feasible domain and l is not, loop finishes here if this is not the case
            # set α as the midpoint of the interval
            alpha.value = np.atleast_1d((u + l)/2.0)
    
            # test the size of the interval against the specified tolerance
            if u-l <= epsilon:
                break
            
            # form and solve problem
            prob = cp.Problem(obj, constraints)
            prob.solve()
    
            # If the problem is feasible u -> α, if not l -> α, best takes the last feasible value as the optimal one as
            # when the tolerance is reached the new α may be out of bounds
            if prob.status == 'optimal':
                u = alpha.value
                best = p.value
            else:
                l = alpha.value
                
            # final condition to check that the interval has converged to order ε, i.e. the range of the optimal sublevel set is <=ε
            if u - l > epsilon and i == (maxLoop-1):
                print("Solution not converged to order epsilon")
        
        return l, u, float(alpha.value), best


Example
-------

As a simple example, we will consider a case with :math:`n=5`, where
:math:`G_{ij} = 0.6` if :math:`i=j` and :math:`0.1` otherwise.

:math:`P_j^{\text{max}} = 1` for all transmitters and the transmitters
are split into two groups, each with :math:`P_l^{\text{gp}} = 1.8`. The
first group contains transmitters 1 & 2, while the second group contains
3,4 & 5.

For all receivers :math:`P_i^{\text{rc}} = 4` and
:math:`\sigma_i = 0.1`.

.. code:: python

    np.set_printoptions(precision=3)
    
    # in this case we will use a gain matrix with a signal weight of 0.6 and interference weight of 0.1
    G = np.array([[0.6,0.1,0.1,0.1,0.1],
                  [0.1,0.6,0.1,0.1,0.1],
                  [0.1,0.1,0.6,0.1,0.1],
                  [0.1,0.1,0.1,0.6,0.1],
                  [0.1,0.1,0.1,0.1,0.6]])
    
    # in this case m=n, but this generalises if we want n receivers and m transmitters
    n, m = np.shape(G)
    
    # set maximum power of each transmitter and receiver saturation level
    P_max = np.array([1.]*n)
    
    # normalised received power, total possible would be all power from all transmitters so 1/n
    P_received = np.array([4.,4.,4.,4.,4.])/n
    
    # set noise level
    sigma = np.array([0.1,0.1,0.1,0.1,0.1])
    
    # group matrix: number of groups by number of transmitters
    Group = np.array([[1.,1.,0,0,0],[0,0,1.,1.,1.]])
    
    # max normalised power for groups, number of groups by 1
    Group_max = np.array([1.8,1.8])
    
    # now run the optimisation problem
    l, u, alpha, best = maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max)
    
    print('Minimum SINR={:.4g}'.format(1/alpha))
    print('Power={}'.format(best))



.. parsed-literal::

    Minimum SINR=1.148
    Power=[0.8 0.8 0.8 0.8 0.8]

