
Derivatives fundamentals
========================

This notebook will introduce you to the fundamentals of computing the
derivative of the solution map to optimization problems. The derivative
can be used for **sensitivity analysis**, to see how a solution would
change given small changes to the parameters, and to compute
**gradients** of scalar-valued functions of the solution.

In this notebook, we will consider a simple disciplined geometric
program. The geometric program under consideration is

.. math::


   \begin{equation}
   \begin{array}{ll}
   \mbox{minimize} & 1/(xyz) \\
   \mbox{subject to} & a(xy + xz + yz) \leq b\\
   & x \geq y^c,
   \end{array}
   \end{equation}

where :math:`x \in \mathbf{R}_{++}`, :math:`y \in \mathbf{R}_{++}`, and
:math:`z \in \mathbf{R}_{++}` are the variables, and
:math:`a \in \mathbf{R}_{++}`, :math:`b \in \mathbf{R}_{++}` and
:math:`c \in \mathbf{R}` are the parameters. The vector

.. math::


   \alpha = \begin{bmatrix} a \\ b \\ c \end{bmatrix}

is the vector of parameters.

.. code:: ipython3

    import cvxpy as cp
    
    
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)
    
    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)
    c = cp.Parameter()
    
    objective_fn = 1/(x*y*z)
    objective = cp.Minimize(objective_fn)
    constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
    problem = cp.Problem(objective, constraints)
    
    problem.is_dgp(dpp=True)




.. parsed-literal::

    True



Notice the keyword argument ``dpp=True``. The parameters must enter in
the DGP problem acording to special rules, which we refer to as ``dpp``.
The DPP rules are described in an `online
tutorial <https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming>`__.

Next, we solve the problem, setting the parameters :math:`a`, :math:`b`
and :math:`c` to :math:`2`, :math:`1`, and :math:`0.5`.

.. code:: ipython3

    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(gp=True, requires_grad=True)
    
    print(x.value)
    print(y.value)
    print(z.value)


.. parsed-literal::

    0.5612147353889386
    0.31496200373359456
    0.36892055859991446


Notice the keyword argument ``requires_grad=True``; this is necessary to
subsequently compute derivatives.

Solution map
------------

The **solution map** of the above problem is a function

.. math:: \mathcal{S} : \mathbf{R}^2_{++} \times \mathbf{R} \to \mathbf{R}^3_{++}

which maps the parameter vector to the vector of optimal solutions

.. math::


   \mathcal S(\alpha) = \begin{bmatrix} x(\alpha) \\ y(\alpha) \\ z(\alpha)\end{bmatrix}.

Here, :math:`x(\alpha)`, :math:`y(\alpha)`, and :math:`z(\alpha)` are
the optimal values of the variables corresponding to the parameter
vector.

As an example, we just saw that

.. math::


   \mathcal S((2.0, 1.0, 0.5)) = \begin{bmatrix} 0.5612 \\ 0.3150 \\ 0.3690 \end{bmatrix}.

Sensitivity analysis
--------------------

When the solution map is differentiable, we can use its derivative

.. math::


   \mathsf{D}\mathcal{S}(\alpha) \in \mathbf{R}^{3 \times 3}

to perform a **sensitivity analysis**, which studies how the solution
would change given small changes to the parameters.

Suppose we perturb the parameters by a vector of small magnitude
:math:`\mathsf{d}\alpha \in \mathbf{R}^3`. We can approximate the change
:math:`\Delta` in the solution due to the perturbation using the
derivative, as

.. math::


   \Delta = \mathcal{S}(\alpha + \mathsf{d}\alpha) - \mathcal{S}(\alpha) \approx \mathsf{D}\mathcal{S}(\alpha) \mathsf{d}\alpha.

We can compute this in CVXPY, as follows.

Partition the perturbation as

.. math::


   \mathsf{d}\alpha = \begin{bmatrix} \mathsf{d}a \\ \mathsf{d}b \\ \mathsf{d}c\end{bmatrix}.

We set the ``delta`` attributes of the parameters to their
perturbations, and then call the ``derivative`` method.

.. code:: ipython3

    da, db, dc = 1e-2, 1e-2, 1e-2
    
    a.delta = da
    b.delta = db
    c.delta = dc
    problem.derivative()

The ``derivative`` method populates the ``delta`` attributes of the
variables as a side-effect, with the predicted change in the variable.
We can compare the predictions to the actual solution of the perturbed
problem.

.. code:: ipython3

    x_hat = x.value + x.delta
    y_hat = y.value + y.delta
    z_hat = z.value + z.delta
    
    a.value += da
    b.value += db
    c.value += dc
    
    problem.solve(gp=True)
    print('x: predicted {0:.5f} actual {1:.5f}'.format(x_hat, x.value))
    print('y: predicted {0:.5f} actual {1:.5f}'.format(y_hat, y.value))
    print('z: predicted {0:.5f} actual {1:.5f}'.format(z_hat, z.value))
    
    a.value -= da
    b.value -= db
    c.value -= dc


.. parsed-literal::

    x: predicted 0.55729 actual 0.55734
    y: predicted 0.31783 actual 0.31783
    z: predicted 0.37179 actual 0.37175


In this case, the predictions and the actual solutions are fairly close.

Gradient
--------

We can compute gradient of a scalar-valued function of the solution with
respect to the parameters. Let
:math:`f : \mathbf{R}^{3} \to \mathbf{R}`, and suppose we want to
compute the gradient of the composition :math:`f \circ \mathcal S`. By
the chain rule,

.. math::


   \nabla f(S(\alpha)) = \mathsf{D}^T\mathcal{S}(\alpha) \begin{bmatrix}\mathsf{d}x \\ \mathsf{d}y \\ \mathsf{d}z\end{bmatrix},

where :math:`\mathsf{D}^T\mathcal{S}` is the adjoint (or transpose) of
the derivative operator, and :math:`\mathsf{d}x`, :math:`\mathsf{d}y`,
and :math:`\mathsf{d}z` are the partial derivatives of :math:`f` with
respect to its arguments.

We can compute the gradient in CVXPY, using the ``backward`` method. As
an example, suppose

.. math::


   f(x, y, z) = \frac{1}{2}(x^2 + y^2 + z^2),

so that :math:`\mathsf{d}x = x`, :math:`\mathsf{d}y = y`, and
:math:`\mathsf{d}z = z`. Let
:math:`\mathsf{d}\alpha = \nabla f(S(\alpha))`, and suppose we subtract
:math:`\eta \mathsf{d}\alpha` from the parameter, where :math:`\eta` is
a positive constant. Using the following code, we can compare
:math:`f(\mathcal S(\alpha - \eta \mathsf{d}\alpha))` with the value
predicted by the gradient,

.. math::


   f(\mathcal S(\alpha - \eta \mathsf{d}\alpha)) \approx f(\mathcal S(\alpha)) - \eta \mathsf{d}\alpha^T\mathsf{d}\alpha.

.. code:: ipython3

    problem.solve(gp=True, requires_grad=True)
    
    def f(x, y, z):
        return 1/2*(x**2 + y**2 + z**2)
    
    original = f(x, y, z).value
    
    x.gradient = x.value
    y.gradient = y.value
    z.gradient = z.value
    problem.backward()
    
    eta = 0.5
    dalpha = cp.vstack([a.gradient, b.gradient, c.gradient])
    predicted = float((original - eta*dalpha.T @ dalpha).value)
    
    a.value -= eta*a.gradient
    b.value -= eta*b.gradient
    c.value -= eta*c.gradient
    problem.solve(gp=True)
    actual = f(x, y, z).value
    
    print('original {0:.5f} predicted {1:.5f} actual {2:.5f}'.format(
           original, predicted, actual))


.. parsed-literal::

    original 0.27513 predicted 0.22709 actual 0.22942

