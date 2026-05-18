
Maximizing the volume of a box
==============================

*This example is adapted from Boyd, Kim, Vandenberghe, and Hassibi,* "`A
Tutorial on Geometric
Programming <https://web.stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`__\ ".

In this example, we maximize the shape of a box with height :math:`h`,
width :math:`w`, and depth :math:`d`, with limits on the wall area
:math:`2(hw + hd)` and the floor area :math:`wd`, subject to bounds on
the aspect ratios :math:`h/w` and :math:`w/d`. The optimization problem
is

.. math::


   \begin{array}{ll}
   \mbox{maximize} & hwd \\
   \mbox{subject to} & 2(hw + hd) \leq A_{\text wall}, \\
   & wd \leq A_{\text flr}, \\
   & \alpha \leq h/w \leq \beta, \\
   & \gamma \leq d/w \leq \delta.
   \end{array}

.. code:: python

    import cvxpy as cp
    
    # Problem data.
    A_wall = 100
    A_flr = 10
    alpha = 0.5
    beta = 2
    gamma = 0.5
    delta = 2
    
    h = cp.Variable(pos=True, name="h")
    w = cp.Variable(pos=True, name="w")
    d = cp.Variable(pos=True, name="d")
    
    volume = h * w * d
    wall_area = 2 * (h * w + h * d)
    flr_area = w * d
    hw_ratio = h/w
    dw_ratio = d/w
    constraints = [
        wall_area <= A_wall,
        flr_area <= A_flr,
        hw_ratio >= alpha,
        hw_ratio <= beta,
        dw_ratio >= gamma,
        dw_ratio <= delta
    ]
    problem = cp.Problem(cp.Maximize(volume), constraints)
    print(problem)


.. parsed-literal::

    maximize h * w * d
    subject to 2.0 * (h * w + h * d) <= 100.0
               w * d <= 10.0
               0.5 <= h / w
               h / w <= 2.0
               0.5 <= d / w
               d / w <= 2.0


.. code:: python

    assert not problem.is_dcp()
    assert problem.is_dgp()
    problem.solve(gp=True)
    problem.value




.. parsed-literal::

    77.45966630736292



.. code:: python

    h.value




.. parsed-literal::

    7.7459666715289766



.. code:: python

    w.value




.. parsed-literal::

    3.872983364643079



.. code:: python

    d.value




.. parsed-literal::

    2.581988871583608



.. code:: python

    # A 1% increase in allowed wall space should yield approximately
    # a 0.83% increase in maximum value.
    constraints[0].dual_value




.. parsed-literal::

    0.8333333206334043



.. code:: python

    # A 1% increase in allowed wall space should yield approximately
    # a 0.66% increase in maximum value.
    constraints[1].dual_value




.. parsed-literal::

    0.6666666801983365


