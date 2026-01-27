
Total variation in-painting
===========================

Grayscale Images
----------------

A grayscale image is represented as an :math:`m \times n` matrix of
intensities :math:`U^\mathrm{orig}` (typically between the values
:math:`0` and :math:`255`). We are given the values
:math:`U^\mathrm{orig}_{ij}`, for :math:`(i,j) \in \mathcal K`, where
:math:`\mathcal K \subset \{1,\ldots, m\} \times \{1, \ldots, n\}` is
the set of indices corresponding to known pixel values. Our job is to
*in-paint* the image by guessing the missing pixel values, *i.e.*, those
with indices not in :math:`\mathcal K`. The reconstructed image will be
represented by :math:`U \in {\bf R}^{m \times n}`, where :math:`U`
matches the known pixels, *i.e.*, :math:`U_{ij} = U^\mathrm{orig}_{ij}`
for :math:`(i,j) \in \mathcal K`.

The reconstruction :math:`U` is found by minimizing the total variation
of :math:`U`, subject to matching the known pixel values. We will use
the :math:`\ell_2` total variation, defined as

.. math::

   \mathop{\bf tv}(U) =
   \sum_{i=1}^{m-1} \sum_{j=1}^{n-1}
   \left\| \left[ \begin{array}{c}
    U_{i+1,j}-U_{ij}\\ U_{i,j+1}-U_{ij} \end{array} \right] \right\|_2.

Note that the norm of the discretized gradient is *not* squared.

We load the original image and the corrupted image and construct the
Known matrix. Both images are displayed below. The corrupted image has
the missing pixels whited out.

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the images.
    u_orig = plt.imread("data/loki512.png")
    u_corr = plt.imread("data/loki512_corrupted.png")
    rows, cols = u_orig.shape
    
    # known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    known = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
             if u_orig[i, j] == u_corr[i, j]:
                known[i, j] = 1
    
    %matplotlib inline
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(u_orig, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(u_corr, cmap='gray');
    ax[1].set_title("Corrupted Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_2_0.png


The total variation in-painting problem can be easily expressed in
CVXPY. We use the solver SCS, which scales to larger problems than ECOS
does.

.. code:: python

    # Recover the original image using total variation in-painting.
    import cvxpy as cp
    
    
    U = cp.Variable(shape=(rows, cols))
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.multiply(known, U) == cp.multiply(known, u_corr)]
    prob = cp.Problem(obj, constraints)
    
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=cp.SCS)
    print("optimal objective value: {}".format(obj.value))


.. parsed-literal::

    ----------------------------------------------------------------------------
    	SCS v2.0.2 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012-2017
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 1554199, CG tol ~ 1/iter^(2.00)
    eps = 1.00e-05, alpha = 1.50, max_iters = 5000, normalize = 1, scale = 1.00
    acceleration_lookback = 20, rho_x = 1.00e-03
    Variables n = 523265, constraints m = 1045507
    Cones:	primal zero / dual free vars: 262144
    	soc vars: 783363, soc blks: 261121
    Setup time: 1.23e-01s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 5.19e+00  4.79e+00  1.00e+00 -5.21e+05  1.51e+04  0.00e+00  1.38e+00 
       100| 4.46e-03  4.69e-03  3.82e-04  1.10e+04  1.10e+04  3.50e-12  3.95e+01 
       200| 3.59e-04  3.83e-04  9.12e-05  1.10e+04  1.10e+04  4.20e-11  7.82e+01 
       300| 7.10e-05  6.96e-05  2.77e-05  1.10e+04  1.10e+04  3.75e-11  1.14e+02 
       400| 3.30e-05  3.39e-05  2.14e-06  1.10e+04  1.10e+04  6.65e-12  1.47e+02 
       500| 2.77e-05  2.85e-05  1.35e-05  1.10e+04  1.10e+04  2.07e-11  1.81e+02 
       600| 1.10e-05  1.09e-05  6.45e-06  1.10e+04  1.10e+04  1.48e-11  2.15e+02 
       700| 1.00e-05  9.49e-06  1.94e-07  1.10e+04  1.10e+04  2.40e-11  2.48e+02 
       720| 9.04e-06  8.24e-06  6.85e-07  1.10e+04  1.10e+04  1.09e-11  2.55e+02 
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 2.55e+02s
    	Lin-sys: avg # CG iterations: 9.58, avg solve time: 1.41e-01s
    	Cones: avg projection time: 3.42e-03s
    	Acceleration: avg step time: 1.71e-01s
    ----------------------------------------------------------------------------
    Error metrics:
    dist(s, K) = 2.1720e-04, dist(y, K*) = 3.7180e-04, s'y/|s||y| = -9.9097e-11
    primal res: |Ax + s - b|_2 / (1 + |b|_2) = 9.0439e-06
    dual res:   |A'y + c|_2 / (1 + |c|_2) = 8.2388e-06
    rel gap:    |c'x + b'y| / (1 + |c'x| + |b'y|) = 6.8544e-07
    ----------------------------------------------------------------------------
    c'x = 11044.2661, -b'y = 11044.2813
    ============================================================================
    optimal objective value: 11044.28989542425


After solving the problem, the in-painted image is stored in
``U.value``. We display the in-painted image and the intensity
difference between the original and in-painted images. The intensity
difference is magnified by a factor of 10 so it is more visible.

.. code:: python

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Display the in-painted image.
    ax[0].imshow(U.value, cmap='gray');
    ax[0].set_title("In-Painted Image")
    ax[0].axis('off')
    
    img_diff = 10*np.abs(u_orig - U.value)
    ax[1].imshow(img_diff, cmap='gray');
    ax[1].set_title("Difference Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_6_0.png


Color Images
============

For color images, the in-painting problem is similar to the grayscale
case. A color image is represented as an :math:`m \times n \times 3`
matrix of RGB values :math:`U^\mathrm{orig}` (typically between the
values :math:`0` and :math:`255`). We are given the pixels
:math:`U^\mathrm{orig}_{ij}`, for :math:`(i,j) \in \mathcal K`, where
:math:`\mathcal K \subset \{1,\ldots, m\} \times \{1, \ldots, n\}` is
the set of indices corresponding to known pixels. Each pixel
:math:`U^\mathrm{orig}_{ij}` is a vector in :math:`{\bf R}^3` of RGB
values. Our job is to *in-paint* the image by guessing the missing
pixels, *i.e.*, those with indices not in :math:`\mathcal K`. The
reconstructed image will be represented by
:math:`U \in {\bf R}^{m \times n \times 3}`, where :math:`U` matches the
known pixels, *i.e.*, :math:`U_{ij} = U^\mathrm{orig}_{ij}` for
:math:`(i,j) \in \mathcal K`.

The reconstruction :math:`U` is found by minimizing the total variation
of :math:`U`, subject to matching the known pixel values. We will use
the :math:`\ell_2` total variation, defined as

.. math::

   \mathop{\bf tv}(U) =
   \sum_{i=1}^{m-1} \sum_{j=1}^{n-1}
   \left\| \left[ \begin{array}{c}
    U_{i+1,j}-U_{ij}\\ 
    U_{i,j+1}-U_{ij} 
    \end{array} \right] \right\|_2.

Note that the norm of the discretized gradient is *not* squared.

We load the original image and construct the Known matrix by randomly
selecting 30% of the pixels to keep and discarding the others. The
original and corrupted images are displayed below. The corrupted image
has the missing pixels blacked out.

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    np.random.seed(1)
    # Load the images.
    u_orig = plt.imread("data/loki512color.png")
    rows, cols, colors = u_orig.shape
    
    # known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    # The known matrix is initialized randomly.
    known = np.zeros((rows, cols, colors))
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > 0.7:
                for k in range(colors):
                    known[i, j, k] = 1        
    u_corr = known * u_orig
    
    # Display the images.
    %matplotlib inline
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(u_orig, cmap='gray');
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(u_corr);
    ax[1].set_title("Corrupted Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_9_0.png


We express the total variation color in-painting problem in CVXPY using
three matrix variables (one for the red values, one for the blue values,
and one for the green values). We use the solver SCS; the solvers ECOS
and CVXOPT don't scale to this large problem.

.. code:: python

    # Recover the original image using total variation in-painting.
    import cvxpy as cp
    
    
    variables = []
    constraints = []
    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        constraints.append(cp.multiply(known[:, :, i], U) == cp.multiply(known[:, :, i], u_corr[:, :, i]))
    
    prob = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
    prob.solve(verbose=True, solver=cp.SCS)
    print("optimal objective value: {}".format(prob.value))


.. parsed-literal::

    WARN: A->p (column pointers) not strictly increasing, column 523264 empty
    WARN: A->p (column pointers) not strictly increasing, column 785408 empty
    WARN: A->p (column pointers) not strictly increasing, column 1047552 empty
    ----------------------------------------------------------------------------
    	SCS v2.0.2 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012-2017
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 3630814, CG tol ~ 1/iter^(2.00)
    eps = 1.00e-05, alpha = 1.50, max_iters = 5000, normalize = 1, scale = 1.00
    acceleration_lookback = 20, rho_x = 1.00e-03
    Variables n = 1047553, constraints m = 2614279
    Cones:	primal zero / dual free vars: 786432
    	soc vars: 1827847, soc blks: 261121
    Setup time: 3.00e-01s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 1.16e+01  1.18e+01  1.00e+00 -1.02e+06  3.34e+04  1.53e-10  3.81e+00 
       100| 2.19e-03  2.32e-03  6.52e-04  1.14e+04  1.15e+04  7.82e-12  1.08e+02 
       200| 4.23e-04  3.78e-04  4.97e-05  1.15e+04  1.15e+04  1.34e-11  2.04e+02 
       300| 9.58e-05  1.10e-04  5.94e-05  1.15e+04  1.15e+04  1.46e-11  2.96e+02 
       400| 4.54e-05  4.57e-05  6.08e-06  1.15e+04  1.15e+04  5.96e-12  3.85e+02 
       500| 2.92e-05  3.19e-05  3.42e-06  1.15e+04  1.15e+04  3.37e-11  4.74e+02 
       600| 1.77e-05  1.87e-05  1.20e-05  1.15e+04  1.15e+04  3.08e-11  5.60e+02 
       700| 1.40e-05  1.43e-05  7.45e-06  1.15e+04  1.15e+04  9.77e-12  6.47e+02 
       760| 9.03e-06  9.70e-06  2.43e-06  1.15e+04  1.15e+04  7.02e-12  6.99e+02 
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 6.99e+02s
    	Lin-sys: avg # CG iterations: 11.66, avg solve time: 4.29e-01s
    	Cones: avg projection time: 4.72e-03s
    	Acceleration: avg step time: 3.94e-01s
    ----------------------------------------------------------------------------
    Error metrics:
    dist(s, K) = 1.8769e-05, dist(y, K*) = 1.1246e-04, s'y/|s||y| = 6.2851e-11
    primal res: |Ax + s - b|_2 / (1 + |b|_2) = 9.0269e-06
    dual res:   |A'y + c|_2 / (1 + |c|_2) = 9.7005e-06
    rel gap:    |c'x + b'y| / (1 + |c'x| + |b'y|) = 2.4293e-06
    ----------------------------------------------------------------------------
    c'x = 11465.6528, -b'y = 11465.5971
    ============================================================================
    optimal objective value: 11465.652787130613


After solving the problem, the RGB values of the in-painted image are
stored in the value fields of the three variables. We display the
in-painted image and the difference in RGB values at each pixel of the
original and in-painted image. Though the in-painted image looks almost
identical to the original image, you can see that many of the RGB values
differ.

.. code:: python

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    %matplotlib inline
    
    rec_arr = np.zeros((rows, cols, colors))
    for i in range(colors):
        rec_arr[:, :, i] = variables[i].value
    rec_arr = np.clip(rec_arr, 0, 1)
    
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    ax[0].imshow(rec_arr)
    ax[0].set_title("In-Painted Image")
    ax[0].axis('off')
    
    img_diff = np.clip(10 * np.abs(u_orig - rec_arr), 0, 1)
    ax[1].imshow(img_diff)
    ax[1].set_title("Difference Image")
    ax[1].axis('off')




.. parsed-literal::

    (-0.5, 511.5, 511.5, -0.5)




.. image:: tv_inpainting_files/tv_inpainting_13_1.png

