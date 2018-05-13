.. _install:

Install
=======

Mac OS X and Linux
------------------

CVXPY supports both Python 2 and Python 3 on OS X and Linux. We recommend
using Anaconda for installation, as we find that most users prefer to let Anaconda
manage dependencies and environments for them.

1. Install `Anaconda`_.

2. Install ``cvxpy`` with ``conda``. 

   ::

      conda install -c cvxgrp cvxpy libgcc

3. Test the installation with ``nose``.

  ::

       conda install nose
       nosetests cvxpy

If you may are comfortable with managing your own environment, you can
instead install CVXPY with pip: ``pip install cvxpy``.

Windows
-------

There are two ways to install CVXPY on Windows.
One method uses Python(x,y), while the other uses Anaconda.
Installation with Python(x,y) is less likely to have problems.
Both installation methods use Python 2.

Windows with Python(x,y)
^^^^^^^^^^^^^^^^^^^^^^^^

1. If you have Python installed already, it's probably a good idea to remove it first.
If you uninstall Anaconda, you may need to take `extra steps to remove all traces of the Anaconda install <http://stackoverflow.com/questions/15828294/problems-in-fully-uninstalling-python-2-7-from-windows-7>`_.

2. Download the `latest version of Python(x,y) <https://python-xy.github.io/downloads.html>`_.

3. Install Python(x,y). When prompted to select optional components, make sure to check CVXOPT and CVXPY, as shown below.

  .. image:: files/windows1.png
      :scale: 100%

  .. image:: files/windows2.png
      :scale: 49%

4. To test the CVXPY installation,
open Python(x,y) and launch the interactive console (highlighted button in the picture).
This will bring up a console.

  .. image:: files/windows3.png
      :scale: 100%

5. From the console, run ``nosetests cvxpy``.
If all but one of the tests pass, your installation was successful.


Windows with Anaconda
^^^^^^^^^^^^^^^^^^^^^

1. Download and install the `latest version of Anaconda <https://www.continuum.io/downloads>`_. You must use the Python 2 version.

2. Download the `Visual Studio C++ compiler for Python <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_.

3. Install SCS from the Anaconda prompt by running the following command:

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

Install from pip
----------------

CVXPY can be installed on all platforms with `pip`_. Simply execute:

  ::

      pip install cvxpy

Though installation with `pip`_ is simple in theory, in practice we have found that novices struggle to install all CVXPY dependencies properly when using `pip`_. We therefore recommend Anaconda as the default installation method.

Install from source
-------------------

CVXPY has the following dependencies:

* Python 2.7 or Python 3.4
* `setuptools`_ >= 1.4
* `toolz`_
* `six <https://pythonhosted.org/six/>`_
* `fastcache <https://github.com/pbrady/fastcache>`_
* `multiprocess`_
* `ECOS`_ >= 2
* `SCS`_ >= 1.1.3
* `NumPy`_ >= 1.8
* `SciPy`_ >= 0.15

To test the CVXPY installation, you additionally need `Nose`_.

CVXPY automatically installs `ECOS`_, `SCS`_, `toolz`_, six, fastcache, and
`multiprocess`_. `NumPy`_ and `SciPy`_ will need to be installed manually,
as will `Swig`_ .  Once you’ve installed these dependencies:

1. Remove all existing installations of
   `CVXcanon <https://github.com/cvxgrp/CVXcanon>`_.
2. Clone the `CVXPY git repository`_.
3. Navigate to the top-level of the cloned directory and run

   ::

       python setup.py install

Install with CVXOPT support
---------------------------

CVXPY supports the `CVXOPT`_ solver.
Simply install CVXOPT such that you can ``import cvxopt`` in Python.
See the `CVXOPT`_ website for installation instructions.

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

Install with XPRESS support
---------------------------

CVXPY supports the XPRESS solver.
Simply install XPRESS such that you can ``import xpress`` in Python.
See the `XPRESS <http://www.fico.com/en/products/fico-xpress-optimization-suite>`_ website for installation instructions.

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
.. _pip: https://pip.pypa.io/
