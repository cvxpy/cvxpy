.. _install:

Install Guide
=============

Mac OS X
--------

1. Install the Command Line Tools for Xcode.

   Download from the `Apple developer site <http://developer.apple.com/downloads>`_.

2. Install `Anaconda`_.

   Follow the instructions on the `website`_.

3. Make sure `Anaconda`_ has the latest version of Python 2.

   ::

       conda update python


4. Install ``numpy`` and ``scipy`` using conda from the command-line.

   ::

       conda install numpy scipy

5. Install ``cvxpy`` with ``pip`` from the command-line.

   ::

       pip install cvxpy

Ubuntu
------

1. Install ``ATLAS`` and ``gfortran`` (needed for ``scs``).

   ::

       sudo apt-get install libatlas-base-dev gfortran

2. Install ``pip``.

   ::

       sudo apt-get install python-pip

3. Install ``numpy`` and ``scipy``.

   ::

       sudo apt-get install python-numpy python-scipy

4. Install ``cvxpy``.

   ::

       sudo pip install cvxpy

Install from source
-------------------

CVXPY has the following dependencies:

* Python 2.7
* `setuptools`_ >= 1.4
* `toolz`_
* `CVXOPT`_ >= 1.1.6
* `ECOS`_ >= 1.0.3
* `SCS`_ >= 1.0.1
* `NumPy`_ >= 1.7.1
* `SciPy`_ >= 0.13.2

To test the CVXPY installation, you additionally need `Nose`_.

CVXPY automatically installs `ECOS`_, `CVXOPT`_, `SCS`_, and `toolz`_.
`NumPy`_ and `SciPy`_ will need to be installed manually. Once you’ve
installed `NumPy`_ and `SciPy`_, installing CVXPY from source is simple:

1. Clone the `CVXPY git repository`_.
2. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _website: https://store.continuum.io/cshop/anaconda/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _toolz: http://github.com/pytoolz/toolz/
.. _CVXOPT: http://cvxopt.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _Nose: http://nose.readthedocs.org
.. _CVXPY git repository: https://github.com/cvxgrp/cvxpy