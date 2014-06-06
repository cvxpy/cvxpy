.. _tv_l2:

Total Variation In-Painting
===========================

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
    orig_img = Image.open("lena512.png")
    corr_img = Image.open("lena512_corrupted.png")

    # Convert to arrays.
    Uorig = np.array(orig_img)
    Ucorr = np.array(corr_img)
    rows, cols = Uorig.shape

    # Known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    Known = np.zeros((rows, cols))
    for i in xrange(rows):
        for j in xrange(cols):
             if Uorig[i, j] == Ucorr[i, j]:
                Known[i, j] = 1

    %matplotlib inline
    plt.figure(); plt.imshow(orig_img);
    plt.title("Original Image")
    plt.figure(); plt.imshow(corr_img);
    plt.title("Corrupted Image")



.. parsed-literal::

    <matplotlib.text.Text at 0x10edb4550>




.. image:: TV_In-Painting_files/TV_In-Painting_3_1.png



.. image:: TV_In-Painting_files/TV_In-Painting_3_2.png


The total variation in-painting problem can be easily expressed in
CVXPY. We use the solver SCS, which finds the optimal value in 9
seconds. The solvers ECOS and CVXOPT take much longer to solve this
large problem (over 2 minutes).

.. code:: python

    # Recover the original image using total variation in-painting.
    from cvxpy import *
    U = Variable(rows, cols)
    obj = Minimize(tv(U))
    constraints = [mul_elemwise(Known, U) == mul_elemwise(Known, Ucorr)]
    prob = Problem(obj, constraints)
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=SCS)

.. parsed-literal::

    ----------------------------------------------------------------------------
        scs v1.0 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
    ----------------------------------------------------------------------------
    Method: sparse-direct, nnz in A = 1547594
    EPS = 1.00e-03, ALPHA = 1.80, MAX_ITERS = 2500, NORMALIZE = 1, SCALE = 5.0
    Variables n = 523265, constraints m = 1045507
    Cones:  primal zero / dual free vars: 262144
        linear vars: 0
        soc vars: 783363, soc blks: 261121
        sd vars: 0, sd blks: 0
        exp vars: 0, dual exp vars: 0
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj |  kappa  | time (s)
    ============================================================================
         0| 1.57e+00  3.56e+00  1.00e+00 -1.45e+07  5.67e+06  2.87e-06  1.79e-01
       100| 2.79e-04  2.78e-03  1.96e-05  2.21e+06  2.21e+06  1.43e-06  8.14e+00
       140| 7.99e-05  8.18e-04  5.88e-06  2.21e+06  2.21e+06  1.43e-06  1.16e+01
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 1.16e+01s, setup time: 2.78e+00s
        Lin-sys: nnz in L factor: 12280804, avg solve time: 5.87e-02s
        Cones: avg projection time: 2.88e-03s
    ----------------------------------------------------------------------------
    Error metrics:
    |Ax + s - b|_2 / (1 + |b|_2) = 7.9862e-05
    |A'y + c|_2 / (1 + |c|_2) = 8.1850e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 5.8838e-06
    dist(s, K) = 0, dist(y, K*) = 0, s'y = 0
    ----------------------------------------------------------------------------
    c'x = 2209248.9113, -b'y = 2209274.9090
    ============================================================================




.. parsed-literal::

    2209248.9112672363



After solving the problem, the in-painted image is stored in
``U.value``. We display the in-painted image and the intensity
difference between the original and in-painted images. The intensity
difference is magnified by a factor of 10 so it is more visible.

.. code:: python

    # Display the in-painted image.
    img_rec = Image.fromarray(U.value)
    plt.figure(); plt.imshow(img_rec)
    plt.title("In-Painted Image")

    img_diff = Image.fromarray(10*np.abs(Uorig - U.value))
    plt.figure(); plt.imshow(img_diff)
    plt.title("Difference Image")



.. parsed-literal::

    <matplotlib.text.Text at 0x10c124690>




.. image:: TV_In-Painting_files/TV_In-Painting_7_1.png



.. image:: TV_In-Painting_files/TV_In-Painting_7_2.png


Total Variation Color In-Painting
---------------------------------

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
    orig_img = Image.open("lena512color.tiff")

    # Convert to arrays.
    Uorig = np.array(orig_img)
    rows, cols, colors = Uorig.shape

    # Known is 1 if the pixel is known,
    # 0 if the pixel was corrupted.
    # The Known matrix is initialized randomly.
    Known = np.zeros((rows, cols, colors))
    for i in xrange(rows):
        for j in xrange(cols):
            if np.random.random() > 0.7:
                for k in xrange(colors):
                    Known[i, j, k] = 1

    Ucorr = Known*Uorig
    corr_img = Image.fromarray(np.uint8(Ucorr))

    # Display the images.
    %matplotlib inline
    plt.figure(); plt.imshow(orig_img);
    plt.title("Original Image")
    plt.figure(); plt.imshow(corr_img);
    plt.title("Corrupted Image")



.. parsed-literal::

    <matplotlib.text.Text at 0x10e5aaa50>




.. image:: TV_In-Painting_files/TV_In-Painting_11_1.png



.. image:: TV_In-Painting_files/TV_In-Painting_11_2.png


We express the total variation color in-painting problem in CVXPY using
three matrix variables (one for the red values, one for the blue values,
and one for the green values). We use the solver SCS, which finds the
optimal value in 25 seconds. The solvers ECOS and CVXOPT don't scale to
this large problem.

.. code:: python

    # Recover the original image using total variation in-painting.
    from cvxpy import *
    variables = []
    constraints = []
    for i in xrange(colors):
        U = Variable(rows, cols)
        variables.append(U)
        constraints.append(mul_elemwise(Known[:, :, i], U) == mul_elemwise(Known[:, :, i], Ucorr[:, :, i]))

    prob = Problem(Minimize(tv(*variables)), constraints)
    prob.solve(verbose=True, solver=SCS)

.. parsed-literal::

    ----------------------------------------------------------------------------
        scs v1.0 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
    ----------------------------------------------------------------------------
    Method: sparse-direct, nnz in A = 3630814
    EPS = 1.00e-03, ALPHA = 1.80, MAX_ITERS = 2500, NORMALIZE = 1, SCALE = 5.0
    Variables n = 1047553, constraints m = 2614279
    Cones:  primal zero / dual free vars: 786432
        linear vars: 0
        soc vars: 1827847, soc blks: 261121
        sd vars: 0, sd blks: 0
        exp vars: 0, dual exp vars: 0
    ----------------------------------------------------------------------------
     Iter | pri res | dua res | rel gap | pri obj | dua obj |  kappa  | time (s)
    ============================================================================
         0| 1.07e+00  2.76e+00  1.00e+00 -1.07e+07  9.06e+06  7.80e-06  5.08e-01
       100| 5.67e-05  6.34e-04  1.46e-05  2.91e+06  2.91e+06  0.00e+00  2.06e+01
    ----------------------------------------------------------------------------
    Status: Solved
    Timing: Solve time: 2.07e+01s, setup time: 8.34e+00s
        Lin-sys: nnz in L factor: 35251632, avg solve time: 1.53e-01s
        Cones: avg projection time: 3.80e-03s
    ----------------------------------------------------------------------------
    Error metrics:
    |Ax + s - b|_2 / (1 + |b|_2) = 5.6668e-05
    |A'y + c|_2 / (1 + |c|_2) = 6.3418e-04
    |c'x + b'y| / (1 + |c'x| + |b'y|) = 1.4607e-05
    dist(s, K) = 0, dist(y, K*) = 0, s'y = 0
    ----------------------------------------------------------------------------
    c'x = 2906814.7944, -b'y = 2906899.7140
    ============================================================================




.. parsed-literal::

    2906814.7944150376



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
    for i in xrange(colors):
        rec_arr[:, :, i] = variables[i].value

    img_rec = Image.fromarray(rec_arr)
    plt.figure(); plt.imshow(img_rec)
    plt.title("In-Painted Image")

    img_diff = Image.fromarray(np.abs(Uorig - rec_arr))
    plt.figure(); plt.imshow(img_diff)
    plt.title("Difference Image")




.. parsed-literal::

    <matplotlib.text.Text at 0x10ce515d0>




.. image:: TV_In-Painting_files/TV_In-Painting_15_1.png



.. image:: TV_In-Painting_files/TV_In-Painting_15_2.png

