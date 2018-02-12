.. _install:

Install Guide
=============

Mac OS X and Linux
------------------

CVXPY supports both Python 2 and Python 3 on OS X and Linux.

1. Install `Anaconda`_.

2. Install ``cvxpy`` with ``conda``.

   ::

      conda install numpy scipy pip
      conda install -c cvxgrp ecos scs multiprocess

3. Clone the `CVXPY git repository`_.
4. Navigate to the top-level of the cloned directory and run

   ::

       git checkout 1.0
       python setup.py install

5. Test the installation with ``nose``.

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
CVXPY supports the `Cbc <https://projects.coin-or.org/Cbc>`_ solver (which includes, amongst other things,
Clp and Cgl) with the help of `cylp <https://github.com/coin-or/CyLP>`_.
Simply install cylp (you will need the Cbc sources which includes `Cgl <https://projects.coin-or.org/Cbc>`_) such you can import this library in Python.
See the `cylp documentation <https://github.com/coin-or/CyLP>`_ for installation instructions.

.. WARNING::
    Python3-support is still not merged into cylp's master-branch! Consider using one of the following two branches: `SteveDiamond/CyLP py3`_ or `jjhelmus/CyLP py3`_.

Install with BONMIN_QP support
------------------------------
CVXPY can use CoinOR's general MINLP-solver `Bonmin`_ to solve MIQP's using a wrapper called `pyMIQP`_.

To use this one you will need:

- Bonmin:
    - Release-version recommended: ships dependent CoinOR projects and scripts to prepare external dependencies
- pyMIQP:
    - Depends on `Eigen`_
    - Depends on `pybind11`_ (modern C++11 compiler needed!)

Install was only tested on Ubuntu-style Linux with Python 2 and 3 (Windows is not expected to work!)

Install:
^^^^^^^^
The following steps are bash-style and related to cvxpy's internal continuous-integration
scripts doing the same for testing!

(Assumption: residing in home-folder with adequate filesystem-rights)

Prepare system
""""""""""""""
::

 sudo apt install wget gfortran gcc g++ autotools-dev automake

Bonmin
""""""
This is a system-wide install.

``./configure`` also allows: ``--enable-cbc-parallel`` (one might need to prepare OpenMP).

::

 wget https://www.coin-or.org/download/source/Bonmin/Bonmin-1.8.6.tgz
 tar -zxvf Bonmin-1.8.6.tgz
 cd Bonmin-1.8.6
 cd ThirdParty
 cd Blas
 ./get.Blas
 cd ..
 cd Lapack
 ./get.Lapack
 cd ..
 cd Mumps
 ./get.Mumps
 cd ..
 cd ..
 ./configure --prefix=/usr --enable-gnu-packages
 make
 sudo make install
 cd ..

pyMIQP (part 1)
"""""""""""""""
::

 wget https://github.com/sschnug/pyMIQP/archive/v0.03.tar.gz
 tar -zxvf v0.03.tar.gz
 cd pyMIQP-0.03
 cd src

Eigen
"""""
This will extract the necessary Eigen-headers into the src-dir of *pyMIQP*.
::

 wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz
 tar -zxvf 3.3.4.tar.gz --strip-components=1 eigen-eigen-5a0156e40feb/Eigen/
 cd ..

pyMIQP (part 2)
"""""""""""""""
Depending on your setup, both of these commands will need ``sudo``.

::

 pip install pybind11
 python setup.py install

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
.. _SteveDiamond/CyLP py3: https://github.com/SteveDiamond/CyLP/tree/py3
.. _jjhelmus/CyLP py3: https://github.com/jjhelmus/CyLP/tree/py3
.. _Bonmin: https://projects.coin-or.org/Bonmin
.. _pyMIQP: https://github.com/sschnug/pyMIQP
.. _Eigen: http://eigen.tuxfamily.org
.. _pybind11: https://github.com/pybind/pybind11
