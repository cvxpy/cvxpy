
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

    ------------------------------------------------------------------
    	       SCS v3.3.1 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012
    ------------------------------------------------------------------
    problem:  variables n: 523265, constraints m: 1045507
    cones: 	  z: primal zero / dual free vars: 262144
    	  q: soc vars: 783363, qsize: 261121
    settings: eps_abs: 1.0e-05, eps_rel: 1.0e-05, eps_infeas: 1.0e-07
    	  alpha: 1.50, scale: 1.00e-01, adaptive_scale: 1
    	  max_iters: 100000, normalize: 1, rho_x: 1.00e-06
    	  acceleration_lookback: 10, acceleration_interval: 5
    lin-sys:  sparse-direct-amd-qdldl
    	  nnz(A): 1554199, nnz(P): 0
    ------------------------------------------------------------------
     iter | pri res | dua res |   gap   |   obj   |  scale  | time (s)
    ------------------------------------------------------------------
         0| 1.56e+01  1.00e+00  4.07e+06 -2.04e+06  1.00e-01  1.34e+00
       250| 9.68e-03  9.95e-03  5.56e-04  1.10e+04  1.00e-01  1.31e+01
       500| 3.04e-03  6.39e-03  3.38e-05  1.10e+04  1.00e-01  2.48e+01
       750| 4.83e-04  2.04e-03  3.59e-05  1.10e+04  1.29e-02  3.72e+01
      1000| 7.01e-05  8.83e-05  3.75e-05  1.10e+04  1.29e-02  4.90e+01
      1250| 1.34e-05  3.57e-05  4.29e-05  1.10e+04  1.29e-02  6.14e+01
      1325| 1.05e-05  1.60e-05  4.38e-05  1.10e+04  1.29e-02  6.53e+01
    ------------------------------------------------------------------
    status:  solved
    timings: total: 6.53e+01s = setup: 1.23e+00s + solve: 6.40e+01s
    	 lin-sys: 3.71e+01s, cones: 5.88e+00s, accel: 4.56e+00s
    ------------------------------------------------------------------
    objective = 11044.275554
    ------------------------------------------------------------------
    optimal objective value: 11044.277093475477


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

    ------------------------------------------------------------------
    	       SCS v3.3.1 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012
    ------------------------------------------------------------------
    problem:  variables n: 1047553, constraints m: 2614279
    cones: 	  z: primal zero / dual free vars: 786432
    	  q: soc vars: 1827847, qsize: 261121
    settings: eps_abs: 1.0e-05, eps_rel: 1.0e-05, eps_infeas: 1.0e-07
    	  alpha: 1.50, scale: 1.00e-01, adaptive_scale: 1
    	  max_iters: 100000, normalize: 1, rho_x: 1.00e-06
    	  acceleration_lookback: 10, acceleration_interval: 5
    lin-sys:  sparse-direct-amd-qdldl
    	  nnz(A): 3630814, nnz(P): 0
    ------------------------------------------------------------------
     iter | pri res | dua res |   gap   |   obj   |  scale  | time (s)
    ------------------------------------------------------------------
         0| 1.60e+01  1.00e+00  4.18e+06 -2.09e+06  1.00e-01  4.33e+00
       250| 2.57e-03  1.10e-02  3.81e-05  1.15e+04  1.00e-01  3.78e+01
       450| 1.28e-05  1.76e-05  4.11e-06  1.15e+04  1.07e-02  6.54e+01
    ------------------------------------------------------------------
    status:  solved
    timings: total: 6.54e+01s = setup: 4.03e+00s + solve: 6.13e+01s
    	 lin-sys: 3.73e+01s, cones: 4.61e+00s, accel: 3.65e+00s
    ------------------------------------------------------------------
    objective = 11465.602408
    ------------------------------------------------------------------
    optimal objective value: 11465.607283681691


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

