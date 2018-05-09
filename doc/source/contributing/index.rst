.. _contributing:

Developer Guide
===============

We welcome all contributors to CVXPY. You don't need to be an expert in convex
to help out!

Deprecated functionality
------------------------

CVXPY is a work in progress. The only APIs guaranteed to be stable are those discussed in the API documentation. All other APIs are subject to change without warning.
Files that are certain to change in the future are marked "DEPRECATED" in the source.

About CVXPY
-----------
CVXPY is a domain-specific language for convex optimization embedded in
Python.  It allows the user to express convex optimization problems in a
natural syntax that follows the math, rather than in the restrictive standard
form required by solvers.  CVXPY makes it easy to combine convex optimization
with high-level features of Python such as parallelism and object-oriented
design.  CVXPY is available at `cvxpy.org <http://www.cvxpy.org/>`_ under the
GPL license, along with documentation and examples.

CVXPY is widely used by researchers and industry practitioners who want to
apply optimization to their problems.  It has been downloaded thousands of
times and used to teach multiple courses.  Many tools have been built on top of
CVXPY, such as an extension for `stochastic optimization
<http://alnurali.github.io/cvxstoc/>`_.

Contact
---------------

The `cvxpy <https://groups.google.com/forum/#!forum/cvxpy>`_ mailing list is
for users and developers of CVXPY.  Use this mailing list to introduce
yourself.  You can also contact the project leads `Steven Diamond
<http://web.stanford.edu/~stevend2/>`_ and `Stephen Boyd
<http://stanford.edu/~boyd/>`_ directly.

Another way to talk with developers is to join the `Gitter chat
<https://gitter.im/cvxgrp/cvxpy>`_. We use GitHub to
track our source code and for tracking and discussing `issues
<https://github.com/cvxgrp/cvxpy/issues>`_.

Getting started
---------------

To get started as a CVXPY developer,
follow the instructions :ref:`here <install>` to install CVXPY from source.

We recommend using `Anaconda`_ as a package manager for Python.
This makes it easy to install dependencies like `NumPy`_ and `SciPy`_.

The CVXPY developers use git for source control. Contributions to CVXPY are
made through pull requests or feature branches.

Our Github `issues <https://github.com/cvxgrp/cvxpy/issues>`_ page is the most
up to date list of CVXPY bugs and feature requests.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _CVXOPT: http://cvxopt.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
