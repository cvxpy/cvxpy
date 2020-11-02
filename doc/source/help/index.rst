.. _help:

Help
====

If you need help with CVXPY, first try using the [CVXPY analyzer](https://github.com/cvxgrp/cvxpyanalyzer). Simply install the package via

  ::

      pip install cvxpyanalyzer

Then pass the CVXPY problem you need help with to the ``tech_support`` function.

.. code:: python

  from analyzer import tech_support

  # Construct CVXPY problem.
  ...

  # Analyze the problem.
  tech_support(problem)

The ``tech_support`` function will guide you through a process of debugging
and analyzing the problem.

If you still need help after using the CVXPY analyzer,
you can post a question on [StackOverflow](https://stackoverflow.com/search?q=cvxpy) or on the `CVXPY mailing list <https://groups.google.com/forum/#!forum/cvxpy>`_.
If you've found a bug in CVXPY or have a feature request,
create an issue on the `CVXPY Github issue tracker <https://github.com/cvxgrp/cvxpy/issues>`_.
