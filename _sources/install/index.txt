.. _install:

Install Guide
=============

Mac OS X and Linux
------------------

CVXPY supports both Python 2 and Python 3 on OS X and Linux.

1. Install `Anaconda`_.

2. Install ``cvxpy`` with ``conda``. 

   ::

      conda install --yes mkl pip nose numpy scipy
      conda install -c conda-forge --yes lapack
      conda install -c cvxgrp --yes ecos scs multiprocess

3. Clone the `CVXPY git repository`_.
4. Remove all existing installations of `CVXcanon`_ if you started at step 3.
5. Navigate to the top-level of the cloned directory and run

   ::

       git checkout 1.0
       python setup.py install

6. Test the installation with ``nose``.

  ::

       conda install nose
       nosetests cvxpy

Install with CVXOPT support
---------------------------

CVXPY supports the `CVXOPT`_ solver.
Simply install CVXOPT by running ``pip install cvxopt``.
If you use Anaconda you will need to run ``conda install nomkl`` first.

Install with Elemental support
------------------------------

CVXPY supports the Elemental solver.
Simply install Elemental such that you can ``import El`` in Python.
See the `Elemental <http://libelemental.org/>`_ website for installation instructions.

Install with GUROBI support
---------------------------

CVXPY supports the GUROBI solver.
Simply install GUROBI such that you can ``import gurobipy`` in Python.
See the `GUROBI <http://www.gurobi.com/>`_ website for installation instructions.

Install with MOSEK support
---------------------------

CVXPY supports the MOSEK solver.
Simply install MOSEK such that you can ``import mosek`` in Python.
See the `MOSEK <https://www.mosek.com/>`_ website for installation instructions.

Install with GLPK support
-------------------------

CVXPY supports the GLPK solver, but only if CVXOPT is installed with GLPK bindings. To install CVXPY and its dependencies with GLPK support, follow these instructions:

1. Install `GLPK <https://www.gnu.org/software/glpk/>`_. We recommend either installing the latest GLPK from source or using a package manager such as apt-get on Ubuntu and homebrew on OS X.

2. Install `CVXOPT`_ with GLPK bindings.

    ::

      CVXOPT_BUILD_GLPK=1
      CVXOPT_GLPK_LIB_DIR=/path/to/glpk-X.X/lib
      CVXOPT_GLPK_INC_DIR=/path/to/glpk-X.X/include
      pip install cvxopt

3. Follow the standard installation procedure to install CVXPY and its remaining dependencies.


Install with Cbc (Clp, Cgl) support
-----------------------------------
CVXPY supports the `Cbc <https://projects.coin-or.org/Cbc>`_ solver (which includes Clp and Cgl) with the help of `cylp <https://github.com/coin-or/CyLP>`_.
Simply install cylp (you will need the Cbc sources which includes `Cgl <https://projects.coin-or.org/Cbc>`_) such you can import this library in Python.
See the `cylp documentation <https://github.com/coin-or/CyLP>`_ for installation instructions.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _website: https://store.continuum.io/cshop/anaconda/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _multiprocess: https://github.com/uqfoundation/multiprocess/
.. _toolz: http://github.com/pytoolz/toolz/
.. _CVXOPT: http://cvxopt.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _Nose: http://nose.readthedocs.org
.. _CVXPY git repository: https://github.com/cvxgrp/cvxpy
.. _CVXcanon: https://github.com/jacklzhu/CVXcanon
.. _Swig: http://www.swig.org/
