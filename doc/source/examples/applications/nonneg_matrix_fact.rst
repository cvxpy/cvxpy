
Nonnegative matrix factorization
================================

| A derivative work by Judson Wilson, 6/2/2014.
| Adapted from the CVX example of the same name, by Argyris Zymnis,
Joelle Skaf and Stephen Boyd

Introduction
------------

We are given a matrix :math:`A \in \mathbf{\mbox{R}}^{m \times n}` and
are interested in solving the problem:

.. raw:: latex

   \begin{array}{ll}
       \mbox{minimize}   & \| A - YX \|_F \\
       \mbox{subject to} & Y \succeq 0 \\
                         & X \succeq 0,
       \end{array}

where :math:`Y \in \mathbf{\mbox{R}}^{m \times k}` and
:math:`X \in \mathbf{\mbox{R}}^{k \times n}`.

This example generates a random matrix :math:`A` and obtains an
*approximate* solution to the above problem by first generating a random
initial guess for :math:`Y` and then alternatively minimizing over
:math:`X` and :math:`Y` for a fixed number of iterations.

Generate problem data
---------------------

.. code:: 

    import cvxpy as cvx
    import numpy as np
    
    # Ensure repeatably random problem data.
    np.random.seed(0)
    
    # Generate random data matrix A.
    m = 10
    n = 10
    k = 5
    A = np.random.rand(m, k).dot(np.random.rand(k, n))
    
    # Initialize Y randomly.
    Y_init = np.random.rand(m, k)

Perform alternating minimization
--------------------------------

.. code:: 

    # Ensure same initial random Y, rather than generate new one
    # when executing this cell.
    Y = Y_init 
    
    # Perform alternating minimization.
    MAX_ITERS = 30
    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1+MAX_ITERS):
        # At the beginning of an iteration, X and Y are NumPy
        # array types, NOT CVXPY variables.
    
        # For odd iterations, treat Y constant, optimize over X.
        if iter_num % 2 == 1:
            X = cvx.Variable(shape=(k, n)
            constraint = [X >= 0]
        # For even iterations, treat X constant, optimize over Y.
        else:
            Y = cvx.Variable(shape=(m, k)
            constraint = [Y >= 0]
        
        # Solve the problem.
        obj = cvx.Minimize(cvx.norm(A - Y*X, 'fro'))
        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.SCS)
    
        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        print 'Iteration {}, residual norm {}'.format(iter_num, prob.value)
        residual[iter_num-1] = prob.value
    
        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            X = X.value
        else:
            Y = Y.value


.. parsed-literal::

    Iteration 1, residual norm 2.76585686659
    Iteration 2, residual norm 0.577758799504
    Iteration 3, residual norm 0.46343315761
    Iteration 4, residual norm 0.300312085357
    Iteration 5, residual norm 0.172468695929
    Iteration 6, residual norm 0.117552622713
    Iteration 7, residual norm 0.0855259222075
    Iteration 8, residual norm 0.0660380454036
    Iteration 9, residual norm 0.0530018181734
    Iteration 10, residual norm 0.0442728793651
    Iteration 11, residual norm 0.0364005958705
    Iteration 12, residual norm 0.0308842140499
    Iteration 13, residual norm 0.0256059616668
    Iteration 14, residual norm 0.0226869576657
    Iteration 15, residual norm 0.0191546943234
    Iteration 16, residual norm 0.0166449632154
    Iteration 17, residual norm 0.0135201384604
    Iteration 18, residual norm 0.0119471133563
    Iteration 19, residual norm 0.0149438374084
    Iteration 20, residual norm 0.0138663023673
    Iteration 21, residual norm 0.00922230392493
    Iteration 22, residual norm 0.00857605731059
    Iteration 23, residual norm 0.0074862441594
    Iteration 24, residual norm 0.00739813239648
    Iteration 25, residual norm 0.0100134191882
    Iteration 26, residual norm 0.00944406772568
    Iteration 27, residual norm 0.008678201611
    Iteration 28, residual norm 0.00873112072225
    Iteration 29, residual norm 0.00798267920957
    Iteration 30, residual norm 0.00846182763828


Output results
--------------

.. code:: 

    #
    # Plot residuals.
    #
    
    import matplotlib.pyplot as plt
    
    # Show plot inline in ipython.
    %matplotlib inline
    
    # Set plot properties.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}
    plt.rc('font', **font)
    
    # Create the plot.
    plt.plot(residual)
    plt.xlabel('Iteration Number')
    plt.ylabel('Residual Norm')
    plt.show()
    
    #
    # Print results.
    #
    print 'Original matrix:'
    print A
    print 'Left factor Y:'
    print Y
    print 'Right factor X:'
    print X
    print 'Residual A - Y * X:'
    print A - Y * X
    print 'Residual after {} iterations: {}'.format(iter_num, prob.value)




.. image:: nonneg_matrix_fact_files/nonneg_matrix_fact_5_0.png


.. parsed-literal::

    Original matrix:
    [[ 1.323426    1.11061189  1.69137835  1.20020115  1.13216889  0.5980743
       1.64965406  0.340611    1.69871738  0.78278448]
     [ 1.73721109  1.40464204  1.90898877  1.60774132  1.53717253  0.62647405
       1.76242265  0.41151492  1.8048194   1.20313124]
     [ 1.4071438   1.10269406  1.75323063  1.18928983  1.23428169  0.60364688
       1.63792853  0.40855006  1.57257432  1.17227344]
     [ 1.3905141   1.33367163  1.07723947  1.67735654  1.33039096  0.42003169
       1.22641711  0.21470465  1.47350799  0.84931787]
     [ 1.42153652  1.13598552  2.00816457  1.11463462  1.17914429  0.69942578
       1.90353699  0.45664487  1.81023916  1.09668578]
     [ 1.60813803  1.23214532  1.73741086  1.3148874   1.27589039  0.40755835
       1.31904948  0.3469129   1.34256526  0.76924618]
     [ 0.90607895  0.6632877   1.25412229  0.81696721  0.87218892  0.50032884
       1.245879    0.25079329  1.25017792  0.72155621]
     [ 1.5691922   1.47359672  1.76518996  1.66268312  1.43746574  0.72486628
       1.97409333  0.39239642  2.09234807  1.16325748]
     [ 1.18723548  1.00282008  1.41532595  1.03836298  0.90382914  0.38460446
       1.213473    0.23641422  1.32784402  0.27179726]
     [ 0.75789915  0.75119989  0.99502166  0.65444815  0.56073096  0.341146
       1.02555143  0.24273668  1.01035919  0.49427978]]
    Left factor Y:
    [[  7.38991833e-01   3.15957978e-01   8.46211348e-01   7.90522539e-01
        8.82326030e-01]
     [  6.37868033e-01   8.22907024e-01   5.32198000e-01   5.70689637e-01
        6.21191813e+00]
     [  5.59748656e-01   6.34112010e-01   7.99615283e-01   1.72054035e-01
        6.92576630e+00]
     [  2.61288516e-01   9.41947419e-01   4.03583183e-02   1.09118729e+00
        9.07778543e-07]
     [  7.89189550e-01   3.41453292e-01   1.17654458e+00   3.93009044e-01
        5.50024762e+00]
     [  7.39615442e-01   4.74493175e-01  -2.23332571e-04   6.74749299e-01
        8.42579458e+00]
     [  4.73914127e-01   3.70454244e-01   8.08948369e-01   1.36848129e-01
        3.44366220e-06]
     [  5.88504809e-01   7.27646377e-01   1.00390505e+00   1.03542480e+00
        3.71366168e-01]
     [  8.14822860e-01   8.87015769e-04   2.91164377e-01   1.17787451e+00
        1.24901335e-06]
     [  4.22680617e-01   7.77641517e-02   5.87259008e-01   6.51086033e-01
        2.66173216e+00]]
    Right factor X:
    [[  1.13055890e+00   4.05899679e-01   1.59181960e+00   6.82867774e-01
        9.75411818e-01   3.23464160e-01   8.83710480e-01   1.64529269e-01
        9.23391090e-01   1.03847861e-01]
     [  9.03465524e-01   6.86715676e-01   6.62169881e-01   1.12490745e+00
        1.03933855e+00   3.06370001e-01   7.29180054e-01   1.18625225e-01
        8.75435486e-01   8.00971786e-01]
     [  6.63783207e-03   1.79085385e-01   3.11550072e-01   2.62447584e-02
        1.60660298e-02   2.77495461e-01   6.46185026e-01   1.51538848e-01
        5.43725876e-01   4.58269799e-01]
     [  2.23736169e-01   5.25847565e-01   2.32705796e-02   4.01864284e-01
        8.80884850e-02   3.23502266e-02   2.59210460e-01   4.94700824e-02
        3.53704441e-01   4.54771630e-02]
     [  2.28693369e-02   2.99295139e-02   2.74091017e-02   5.18043712e-04
        1.64409925e-04   2.93841883e-04   1.72556068e-02   1.61648660e-02
        6.42191908e-04   3.35760370e-02]]
    Residual A - Y * X:
    [[ -1.68636983e-04   3.60207904e-05  -3.98077462e-04  -2.04516172e-04
       -4.16031924e-04   1.58422455e-03  -7.36906450e-04  -5.91624286e-05
       -5.46879309e-04  -4.00508769e-04]
     [ -6.83414765e-04  -6.95866828e-04  -6.36012200e-04  -5.76770312e-05
       -1.33274006e-04   6.27824831e-05  -3.32923224e-04  -3.46132086e-04
        2.01209611e-04  -6.50509595e-04]
     [ -7.73470728e-04  -9.20826799e-04  -6.31154497e-04   2.22522673e-05
        9.78991659e-05  -1.17532606e-03   8.50142570e-05  -4.05047780e-04
        5.07774407e-04  -5.64939614e-04]
     [ -3.10948903e-04  -2.61145596e-04  -3.80145546e-04  -2.41030729e-04
       -2.44643018e-04   4.30377374e-04  -2.61672784e-04  -1.20679505e-04
       -2.79346735e-04  -4.08818401e-04]
     [ -7.07080893e-04  -8.12121976e-04  -6.37850976e-04  -4.46396107e-05
        4.76451453e-05  -1.27571378e-03   9.33258194e-05  -3.50044450e-04
        3.29244878e-04  -4.88627114e-04]
     [ -3.85692817e-04  -8.61465250e-04  -6.94274594e-04   5.50232944e-04
        4.82360881e-04  -1.29338044e-03  -6.97353986e-04  -6.10252918e-04
        2.71414348e-04  -1.10507632e-03]
     [ -3.89135610e-04  -3.02813339e-04  -7.79701822e-04   3.94758648e-04
       -1.51231975e-04   4.63196558e-03  -1.25457666e-03  -4.81723061e-04
        1.10896155e-05  -1.32234306e-03]
     [ -3.69191741e-04  -3.38504643e-04  -4.69486964e-04  -3.62561200e-04
       -2.38561015e-04  -6.07402720e-04  -6.92616183e-05  -1.03315580e-04
       -4.01477226e-04  -2.98251672e-04]
     [ -2.37003094e-04  -5.11457693e-05  -4.34540394e-04  -3.83365150e-05
       -3.15650926e-04   1.86541629e-03  -7.04768572e-04  -1.45497987e-04
       -2.65728261e-04  -5.29103341e-04]
     [ -6.65085219e-04  -9.73758356e-04  -3.70136974e-04  -1.03883251e-04
        3.94287162e-04  -4.20725740e-03   1.14398039e-03  -2.59815037e-04
        6.72891426e-04  -4.62543141e-06]]
    Residual after 30 iterations: 0.00846182763828


