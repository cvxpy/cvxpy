.. _cvxpygen-gradient:

Differentiating Through Problems
=================================

CVXPYgen supports differentiating through quadratic programs.
To enable this feature, set ``gradient=True`` when generating code.
You can use the generated code together with `CVXPYlayers <https://github.com/cvxgrp/cvxpylayers>`_ as

.. code-block:: python

    cpg.generate_code(problem, code_dir='code_diff', gradient=True)

    from code_diff.cpg_solver import forward, backward
    from cvxpylayers.torch import CvxpyLayer

    layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], custom_method=(forward, backward))

See our `manuscript <https://stanford.edu/~boyd/papers/cvxpygen_grad.html>`_ for more details
and `examples/paper_grad <https://github.com/cvxgrp/cvxpygen/tree/master/examples/paper_grad>`_
for three practical examples (from our manuscript), in the areas of machine learning, control, and finance.
