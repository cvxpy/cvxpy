.. _install:

Install
=======

*Note*: Version 1.0 of CVXPY is incompatible with previous versions in minor
ways. See :ref:`updates` for how to update legacy code to a form that's
compatible with 1.0.

Mac OS X, Windows, and Linux
----------------------------

CVXPY supports both Python 2 and Python 3 on OS X, Windows, and Linux. We recommend using
pip for installation. You may want to isolate
your installation in a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_.
If you prefer `Anaconda`_ to `pip`_, see the 
:ref:`Anaconda installation guide <anaconda-installation>`.

1. (Windows only) Download the `Visual Studio C++ compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_
or the Visual Studio build tools for Python 3 (`download <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_, `install instructions <https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view?usp=sharing>`_).

2. Install ``cvxpy``.
  ::

      pip install cvxpy

3. Test the installation with ``nose``.
  ::

      pip install nose
      nosetests cvxpy

Other Platforms
---------------

The CVXPY installation process on other platforms is less automated and less well tested. Check `this page <https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions-for-non-standard-platforms>`_ for instructions for your platform.

.. _anaconda-installation:

Anaconda
----------------

`Anaconda`_ is a system for package and environment management.

1. Install `Anaconda`_.

2. Install `pip`_ and ``setuptools`` with ``conda``.

   ::

      conda install pip
      pip install --upgrade setuptools

3. (Windows only) Download the `Visual Studio C++ compiler for Python 2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_
or the `Visual Studio build tools for Python 3 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_.
   

4. Install ``cvxpy`` with `pip`_ inside `Anaconda`_.

   ::

      pip install cvxpy

5. Test the installation with ``nose``.

  ::

       conda install nose
       nosetests cvxpy

Install from source
-------------------

CVXPY has the following dependencies:

* Python 2.7, 3.4, 3.5, 3.6, or 3.7.
* `six <https://pythonhosted.org/six/>`_
* `multiprocess`_
* `OSQP`_
* `ECOS`_ >= 2
* `SCS`_ >= 1.1.3
* `NumPy`_ >= 1.15
* `SciPy`_ >= 1.1.0

To test the CVXPY installation, you additionally need `Nose`_.

CVXPY automatically installs `OSQP`_, `ECOS`_, `SCS`_, six, and
`multiprocess`_. `NumPy`_ and `SciPy`_ will need to be installed manually,
as will `Swig`_ . Once you’ve installed these dependencies:

1. Clone the `CVXPY git repository`_.
2. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

Install with CVXOPT and GLPK support
------------------------------------

CVXPY supports the `CVXOPT`_ solver.
Additionally, through CVXOPT, CVXPY supports the `GLPK`_ solver. On `most
platforms <http://cvxopt.org/install/index.html#installing-a-pre-built-package>`_,
`CVXOPT`_ comes with GLPK bundled. On such platforms, installing CVXOPT with

  ::

      pip install cvxopt

should suffice to get support for both CVXOPT and GLPK.

On other platforms, to install CVXPY and its dependencies with GLPK support, follow these instructions:

1. Install `GLPK <https://www.gnu.org/software/glpk/>`_. We recommend either installing the latest GLPK from source or using a package manager such as apt-get on Ubuntu and homebrew on OS X.

2. Install `CVXOPT`_ with GLPK bindings.

    ::

      CVXOPT_BUILD_GLPK=1
      CVXOPT_GLPK_LIB_DIR=/path/to/glpk-X.X/lib
      CVXOPT_GLPK_INC_DIR=/path/to/glpk-X.X/include
      pip install cvxopt

3. Follow the standard installation procedure to install CVXPY and its remaining dependencies.

Install with GUROBI support
---------------------------

CVXPY supports the GUROBI solver.
Install GUROBI version 7.5.2 or greater such that you can ``import gurobipy`` in Python.
See the `GUROBI <http://www.gurobi.com/>`_ website for installation instructions.

Install with MOSEK support
---------------------------

CVXPY supports the MOSEK solver.
Simply install MOSEK such that you can ``import mosek`` in Python.
See the `MOSEK <https://www.mosek.com/>`_ website for installation instructions.

Install with XPRESS support
---------------------------

CVXPY supports the XPRESS solver.
Simply install XPRESS such that you can ``import xpress`` in Python.
See the `XPRESS <http://www.fico.com/en/products/fico-xpress-optimization-suite>`_ website for installation instructions.

Install with Cbc (Clp, Cgl) support
-----------------------------------
CVXPY supports the `Cbc <https://projects.coin-or.org/Cbc>`_ solver (which includes Clp and Cgl) with the help of `cylp <https://github.com/coin-or/CyLP>`_.
Simply install cylp (you will need the Cbc sources which includes `Cgl <https://projects.coin-or.org/Cbc>`_) such you can import this library in Python.
See the `cylp documentation <https://github.com/coin-or/CyLP>`_ for installation instructions.

Install with CPLEX support
--------------------------

CVXPY supports the CPLEX solver.
Simply install CPLEX such that you can ``import cplex`` in Python.
See the `CPLEX <https://www.ibm.com/support/knowledgecenter/SSSA5P>`_ website for installation instructions.

Install with SDPT3 support
--------------------------

The `sdpt3glue package <https://github.com/TrishGillett/pysdpt3glue>`_ allows you to model problems with CVXPY and solve them with SDPT3.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _website: https://store.continuum.io/cshop/anaconda/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _multiprocess: https://github.com/uqfoundation/multiprocess/
.. _CVXOPT: http://cvxopt.org/
.. _OSQP: https://osqp.org/
.. _ECOS: http://github.com/ifa-ethz/ecos
.. _SCS: http://github.com/cvxgrp/scs
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/
.. _Nose: http://nose.readthedocs.org
.. _CVXPY git repository: https://github.com/cvxgrp/cvxpy
.. _Swig: http://www.swig.org/
.. _pip: https://pip.pypa.io/
.. _GLPK: https://www.gnu.org/software/glpk/
