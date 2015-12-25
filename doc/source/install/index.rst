.. _install:

Install Guide
=============

Mac OS X
--------

The following instructions assume you already have Python installed.
CVXPY supports both Python 2 and Python 3.

1. Install the Command Line Tools for Xcode.

   Download from the `Apple developer site <http://developer.apple.com/downloads>`_.

2. If you don't have ``pip`` installed, follow the instructions `here <https://pip.pypa.io/en/latest/installing.html>`_ to install it.

3. Install ``numpy`` with ``pip`` from the command-line.

   ::

     pip install numpy

4. Install ``cvxpy`` with ``pip`` from the command-line.

   ::

       pip install cvxpy

5. Test the installation with ``nose``.

  ::

       pip install nose
       nosetests cvxpy

Ubuntu 14.04
------------

The following instructions are for installing CVXPY with Python 2.
To install CVXPY with Python 3, simply install the Python 3 version of all the packages.

1. Make sure ``apt-get`` is up-to-date.

  ::

      sudo apt-get update

2. Install ``ATLAS`` and ``gfortran`` (needed for ``SCS``).

   ::

       sudo apt-get install libatlas-base-dev gfortran

3. Install ``python-dev``.

   ::

       sudo apt-get install python-dev

4. Install ``pip``.

   ::

       sudo apt-get install python-pip

5. Install ``numpy`` and ``scipy``.

   ::

       sudo apt-get install python-numpy python-scipy

6. Install ``cvxpy``.

   ::

       sudo pip install cvxpy

  or to install locally

   ::

      pip install --user cvxpy

7. Install ``nose``.

  ::

       sudo apt-get install python-nose

8. Test the installation with ``nose``.

  ::

       nosetests cvxpy

Windows
-------

There are two ways to install CVXPY on Windows.
One method uses Python(x,y), while the other uses Anaconda.
Follow the Python(x,y) method unless you have a strong preference for Anaconda.

Windows with Python(x,y)
^^^^^^^^^^^^^^^^^^^^^^^^

1. If you have Python installed already, it's probably a good idea to remove it first. (Sorry!)

2. Download the latest version of `Python(x,y) <https://python-xy.github.io/downloads.html>`_.

3. Install Python(x,y). When prompted to select optional components, make sure to check CVXOPT and CVXPY, as shown below.

  .. image:: files/windows1.png
      :scale: 100%

  .. image:: files/windows2.png
      :scale: 49%

4. To test the CVXPY installation,
open Python(x,y) and launch the interactive console (highlighted button in the picture).
This will bring up a console.

  .. image:: files/windows4.png
      :scale: 100%

5. From the console, run ``nosetests cvxpy``.
If all the tests pass, your installation was successful.


Windows with Anaconda
^^^^^^^^^^^^^^^^^^^^^

1. Download and install the latest version of `Anaconda <https://www.continuum.io/downloads>`_.

2. Download the `Visual Studio C++ compiler for Python <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_.

3. Open the Anaconda prompt and install CVXOPT by running the following command:

  ::

      conda install -c https://conda.anaconda.org/omnia cvxopt

4. Install SCS from the Anaconda prompt by running the following command:

  ::

      conda install -c https://conda.anaconda.org/omnia scs

4. Install CVXPY from the Anaconda prompt by running the following command:

  ::

      pip install cvxpy

5. From the console, run ``nosetests cvxpy``.
If all the tests pass, your installation was successful.

Other Platforms
---------------

The CVXPY installation process on other platforms is less automated and less well tested. Check `this page <https://github.com/cvxgrp/cvxpy/wiki/CVXPY-installation-instructions-for-non-standard-platforms>`_ for instructions for your platform.

Install from source
-------------------

CVXPY has the following dependencies:

* Python 2.7 or Python 3.4
* `setuptools`_ >= 1.4
* `toolz`_
* `multiprocess`_
* `CVXOPT`_ >= 1.1.6
* `ECOS`_ >= 2
* `SCS`_ >= 1.0.1
* `NumPy`_ >= 1.8
* `SciPy`_ >= 0.13
* `CVXcanon`_ >= 0.0.17

To test the CVXPY installation, you additionally need `Nose`_.

CVXPY automatically installs `ECOS`_, `CVXOPT`_, `SCS`_, `toolz`_, and
`multiprocess`_. `NumPy`_ and `SciPy`_ will need to be installed manually. You
may also need to install `Swig`_ to build `CVXcanon`_. Once you’ve installed
`NumPy`_, `SciPy`_, and `Swig`_, installing CVXPY from source is simple:

1. Clone the `CVXPY git repository`_.
2. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

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


Install with GUROBI support
---------------------------

CVXPY supports the GUROBI solver.
Simply install GUROBI such that you can ``import gurobipy`` in Python.
See the `GUROBI <http://www.gurobi.com/>`_ website for installation instructions.

Install with Elemental support
------------------------------

CVXPY supports the Elemental solver.
Simply install Elemental such that you can ``import El`` in Python.
See the `Elemental <http://libelemental.org/>`_ website for installation instructions.

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
