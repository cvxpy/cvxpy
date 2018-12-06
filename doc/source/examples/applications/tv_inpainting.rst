
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

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the images.
    orig_img = Image.open("data/lena512.png")
    corr_img = Image.open("data/lena512_corrupted.png")
    
    
    # Convert to arrays.
    Uorig = np.array(orig_img)
    Ucorr = np.array(corr_img)
    rows, cols = Uorig.shape
    
    # Known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    Known = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
             if Uorig[i, j] == Ucorr[i, j]:
                Known[i, j] = 1
    
    %matplotlib inline
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    ax[0].imshow(orig_img);
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(corr_img);
    ax[1].set_title("Corrupted Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_2_0.png


The total variation in-painting problem can be easily expressed in
CVXPY. We use the solver SCS, which finds the optimal value in a few
seconds. The solvers ECOS and CVXOPT take much longer to solve this
large problem.

.. code:: python

    # Recover the original image using total variation in-painting.
    import cvxpy as cp
    U = cp.Variable(shape=(rows, cols))
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.multiply(Known, U) == cp.multiply(Known, Ucorr)]
    prob = cp.Problem(obj, constraints)
    
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=cp.SCS)
    print("optimal objective value: {}".format(obj.value))


.. parsed-literal::

    ----------------------------------------------------------------------------
    	SCS v1.2.6 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012-2016
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 1547594, CG tol ~ 1/iter^(2.00)
    eps = 1.00e-03, alpha = 1.50, max_iters = 2500, normalize = 1, scale = 1.00
    Variables n = 523265, constraints m = 1045507
    Cones:	primal zero / dual free vars: 262144
    	soc vars: 783363, soc blks: 261121
    Setup time: 2.24e-01s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 5.33e+00  4.95e+00  1.00e+00 -1.49e+08  6.22e+06  3.51e-08  1.78e+00 
       100| 3.43e-03  1.35e-03  5.41e-03  2.16e+06  2.18e+06  4.13e-09  3.87e+01 
       200| 1.09e-03  2.52e-04  1.18e-03  2.20e+06  2.20e+06  4.25e-09  7.21e+01 
       220| 9.27e-04  1.95e-04  9.53e-04  2.20e+06  2.21e+06  4.26e-09  7.86e+01 
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 7.87e+01s
    	Lin-sys: avg # CG iterations: 9.54, avg solve time: 3.04e-01s
    	Cones: avg projection time: 5.71e-03s
    ----------------------------------------------------------------------------
    Error metrics:
    dist(s, K) = 4.9738e-14, dist(y, K*) = 2.2204e-16, s'y/|s||y| = 1.6244e-17
    |Ax + s - b|_2 / (1 + |b|_2) = 9.2692e-04
    |A'y + c|_2 / (1 + |c|_2) = 1.9490e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 9.5346e-04
    ----------------------------------------------------------------------------
    c'x = 2201748.4016, -b'y = 2205950.9682
    ============================================================================
    optimal objective value: 2199728.631919451


After solving the problem, the in-painted image is stored in
``U.value``. We display the in-painted image and the intensity
difference between the original and in-painted images. The intensity
difference is magnified by a factor of 10 so it is more visible.

.. code:: python

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Display the in-painted image.
    img_rec = Image.fromarray(U.value)
    ax[0].imshow(img_rec);
    ax[0].set_title("In-Painted Image")
    ax[0].axis('off')
    
    img_diff = Image.fromarray(10*np.abs(Uorig - U.value))
    ax[1].imshow(img_diff);
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

    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    np.random.seed(1)
    # Load the images.
    orig_img = Image.open("data/lena512color.tiff")
    
    # Convert to arrays.
    Uorig = np.array(orig_img)
    rows, cols, colors = Uorig.shape
    
    # Known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    # The Known matrix is initialized randomly.
    Known = np.zeros((rows, cols, colors))
    for i in range(rows):
        for j in range(cols):
            if np.random.random() > 0.7:
                for k in range(colors):
                    Known[i, j, k] = 1
                
    Ucorr = Known*Uorig
    corr_img = Image.fromarray(np.uint8(Ucorr))
    
    # Display the images.
    %matplotlib inline
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    ax[0].imshow(orig_img);
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(corr_img);
    ax[1].set_title("Corrupted Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_9_0.png


We express the total variation color in-painting problem in CVXPY using
three matrix variables (one for the red values, one for the blue values,
and one for the green values). We use the solver SCS, which finds the
optimal value in 25 seconds. The solvers ECOS and CVXOPT don't scale to
this large problem.

.. code:: python

    # Recover the original image using total variation in-painting.
    import cvxpy as cp
    variables = []
    constraints = []
    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        constraints.append(cp.multiply(Known[:, :, i], U) == cp.multiply(Known[:, :, i], Ucorr[:, :, i]))
    
    prob = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
    prob.solve(verbose=True, solver=cp.SCS)
    print("optimal objective value: {}".format(obj.value))


.. parsed-literal::

    WARN: A->p (column pointers) not strictly increasing, column 523264 empty
    WARN: A->p (column pointers) not strictly increasing, column 785408 empty
    WARN: A->p (column pointers) not strictly increasing, column 1047552 empty
    ----------------------------------------------------------------------------
    	SCS v1.2.6 - Splitting Conic Solver
    	(c) Brendan O'Donoghue, Stanford University, 2012-2016
    ----------------------------------------------------------------------------
    Lin-sys: sparse-indirect, nnz in A = 3630814, CG tol ~ 1/iter^(2.00)
    eps = 1.00e-03, alpha = 1.50, max_iters = 2500, normalize = 1, scale = 1.00
    Variables n = 1047553, constraints m = 2614279
    Cones:	primal zero / dual free vars: 786432
    	soc vars: 1827847, soc blks: 261121
    Setup time: 4.98e-01s
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj | kap/tau | time (s)
    ----------------------------------------------------------------------------
         0| 1.15e+01  1.16e+01  1.00e+00 -3.08e+08  8.93e+06  4.08e-08  5.82e+00 
       100| 1.48e-03  4.97e-04  7.91e-04  2.90e+06  2.90e+06  4.38e-09  1.07e+02 
       140| 7.64e-04  1.78e-04  3.23e-04  2.90e+06  2.91e+06  4.40e-09  1.46e+02 
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 1.46e+02s
    	Lin-sys: avg # CG iterations: 11.11, avg solve time: 9.13e-01s
    	Cones: avg projection time: 7.61e-03s
    ----------------------------------------------------------------------------
    Error metrics:
    dist(s, K) = 5.6843e-14, dist(y, K*) = 2.2204e-16, s'y/|s||y| = 2.8317e-17
    |Ax + s - b|_2 / (1 + |b|_2) = 7.6373e-04
    |A'y + c|_2 / (1 + |c|_2) = 1.7788e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 3.2341e-04
    ----------------------------------------------------------------------------
    c'x = 2903331.8699, -b'y = 2905210.4273
    ============================================================================
    optimal objective value: 2199728.631919451


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
    
    # Load variable values into a single array.
    rec_arr = np.zeros((rows, cols, colors), dtype=np.uint8)
    for i in range(colors):
        rec_arr[:, :, i] = variables[i].value
    
    fig, ax = plt.subplots(1, 2,figsize=(10, 5))
    # Display the in-painted image.
    img_rec = Image.fromarray(rec_arr)
    ax[0].imshow(img_rec);
    ax[0].set_title("In-Painted Image")
    ax[0].axis('off')
    
    img_diff = Image.fromarray(np.abs(Uorig - rec_arr))
    ax[1].imshow(img_diff);
    ax[1].set_title("Difference Image")
    ax[1].axis('off');



.. image:: tv_inpainting_files/tv_inpainting_13_0.png

