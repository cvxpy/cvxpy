=====================================
FAQ
=====================================

.. contents::
  :local:
  :backlinks: none
  :depth: 1

Where can I get help with CVXPY?
--------------------------------
You can post questions about how to use CVXPY on StackOverflow or on the `CVXPY mailing list <https://groups.google.com/forum/#!forum/cvxpy>`_.
If you've found a bug in CVXPY or have a feature request,
create an issue on the `CVXPY Github issue tracker <https://github.com/cvxpy/cvxpy/issues>`_.

Where can I learn more about convex optimization?
-------------------------------------------------
The book `Convex Optimization <http://web.stanford.edu/~boyd/cvxbook/>`_ by Boyd and Vandenberghe is available for free online and has extensive background on convex optimization.
To learn more about disciplined convex programming,
visit the `DCP tutorial website <http://dcp.stanford.edu/>`_.

How do I know which version of CVXPY I'm using?
-----------------------------------------------
To check which version of CVXPY you have installed,
run the following code snippet in the Python prompt:

.. code:: python

  import cvxpy
  print cvxpy.__version__

What do I do if I get a ``DCPError`` exception?
-----------------------------------------------
The problems you solve in CVXPY must follow the rules of disciplined convex programming (DCP).
DCP is like a type system for optimization problems.
For more about DCP, see the :ref:`DCP tutorial section <dcp>` or the `DCP tutorial website <http://dcp.stanford.edu/>`_.

How do I find DCP errors?
-------------------------
You can test whether a problem, objective, constraint, or expression satisfies the DCP
rules by calling ``object.is_dcp()``.
If the function returns ``False``,
there is a DCP error in that object.

What do I do if I get a ``SolverError`` exception?
--------------------------------------------------
Sometimes solvers encounter numerical issues and fail to solve a problem, in which case CVXPY raises a ``SolverError``.
If this happens to you,
try using different solvers on your problem,
as discussed in the "Choosing a solver" section of :ref:`Advanced Features <advanced>`.
If the solver CVXOPT fails, try using the solver option ``kktsolver=ROBUST_KKTSOLVER``.

What solvers does CVXPY support?
--------------------------------
See the "Solve method options" section in :ref:`Advanced Features <advanced>` for a list of the solvers CVXPY supports.
If you would like to use a solver CVXPY does not support,
make a feature request on the `CVXPY Github issue tracker <https://github.com/cvxpy/cvxpy/issues>`_.

What are the differences between CVXPY's solvers?
-------------------------------------------------
The solvers support different classes of problems and occupy different points on the Pareto frontier of speed, accuracy, and open source vs. closed source.
See the "Solve method options" section in :ref:`Advanced Features <advanced>` for details.

What do I do if I get "Exception: Cannot evaluate the truth value of a constraint"?
-----------------------------------------------------------------------------------
This error likely means you are chaining constraints (e.g., writing an
expression like ``0 <= x <= 1``) or using the built-in Python ``max`` and ``min``
functions on CVXPY expressions.
It is not possible for CVXPY to correctly handle these use cases,
so CVXPY throws an (admittedly cryptic) exception.

Can I use NumPy functions on CVXPY objects?
-------------------------------------------
No, you can only use CVXPY functions on CVXPY objects.
If you use a NumPy function on a CVXPY object,
it will probably fail in a confusing way.

Can I use SciPy sparse matrices with CVXPY?
-------------------------------------------
Yes, they are fully supported.

How do I constrain a CVXPY matrix expression to be positive semidefinite?
-------------------------------------------------------------------------
See :ref:`Advanced Features <advanced>`.

How do I create variables with special properties, such as boolean or symmetric variables?
------------------------------------------------------------------------------------------
See :ref:`Advanced Features <advanced>`.

How do I create a variable that has multiple special properties, such as boolean and symmetric?
-----------------------------------------------------------------------------------------------
Create one variable with each desired property, and then set them all equal by adding equality constraints.

How do I create variables with more than 2 dimensions?
------------------------------------------------------
You must mimic the extra dimensions using a dict,
as described in `this Github issue <https://github.com/cvxpy/cvxpy/issues/198>`__.

Why does it take so long to compile my Problem?
-----------------------------------------------
In general, you should vectorize CVXPY expressions whenever possible if you
care about performance (e.g., write A * x == b instead of a_i  * x == b_i for
every row a_i of A). Consult this `IPython notebook <https://github.com/cvxpy/cvxpy/blob/1.0/examples/notebooks/building_models_with_fast_compile_times.ipynb>`_ for details.

--------------------
How does CVXPY work?
--------------------
The fundamental concepts behind CVXPY are discussed in `this paper <https://web.stanford.edu/~boyd/papers/pdf/cvxpy_paper.pdf>`_.

How do I cite CVXPY?
--------------------
If you use CVXPY for published work, we encourage you to cite the software.
Use the following BibTeX citation:

::

    @article{cvxpy,
      author       = {Steven Diamond and Stephen Boyd},
      title        = {{CVXPY}: A {P}ython-Embedded Modeling Language for Convex Optimization},
      journal      = {Journal of Machine Learning Research},
      note         = {To appear},
      url          = {http://stanford.edu/~boyd/papers/pdf/cvxpy_paper.pdf},
      year         = {2016},
    }
