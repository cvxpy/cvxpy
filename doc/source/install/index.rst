.. _install:

Install
=======

CVXPY supports Python 3 on Linux, macOS, and Windows. You can use
pip or conda for installation. You may want to isolate
your installation in a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_,
or a conda environment.

pip
---

(Windows only) Download the Visual Studio build tools for Python 3
(`download <https://visualstudio.microsoft .com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_,
`install instructions <https://drive.google.com/file/d/0B4GsMXCRaSSIOWpYQkstajlYZ0tPVkNQSElmTWh1dXFaYkJr/view?usp=sharing>`_).

(macOS only) Install the Xcode command line tools.

(optional) Create and activate a virtual environment.

Install CVXPY using `pip`_:

  ::

      pip install cvxpy

You can add solver names as "extras"; `pip` will then install the necessary
additional Python packages.

  ::

      pip install cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]


.. _conda-installation:

conda
-----

`conda`_ is a system for package and environment management.

(Windows only) Download the `Visual Studio build tools for Python 3 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_.

1. Install `conda`_.

2. Create a new conda environment,
  ::

      conda create --name cvxpy_env
      conda activate cvxpy_env

 or activate an existing one

3. Install ``cvxpy`` from `conda-forge <https://conda-forge.org/>`_
   ::

      conda install -c conda-forge cvxpy

.. _install_from_source:

Install from source
-------------------

We strongly recommend using a fresh virtual environment (virtualenv or conda) when installing CVXPY from source.

CVXPY has the following dependencies:

 * Python >= 3.7
 * `OSQP`_ >= 0.4.1
 * `ECOS`_ >= 2
 * `SCS`_ >= 1.1.6
 * `NumPy`_ >= 1.15
 * `SciPy`_ >= 1.1.0

All required packages are installed automatically alongside CVXPY.

Perform the following steps to install CVXPY from source:

 1. Clone the official `CVXPY git repository`_, or a newly minted fork of the CVXPY repository.
 2. Navigate to the top-level of the cloned directory.
 3. If you want to use CVXPY with editable source code, run
    ::

        pip install -e .

    otherwise, run
    ::

        pip install .

Apple M1 users
~~~~~~~~~~~~~~
Apple M1 users have had trouble installing CVXPY using the commands above.
That trouble stemmed partly from a configuration error in CVXPY's
``pyproject.toml``, which has been fixed in CVXPY 1.1.19 and 1.2.0.
If you have those versions (or newer) then the above commands should
work *provided* (1) you have ``cmake`` installed via Homebrew and (2)
you have an ECOS 2.0.5 wheel. The cmake requirement stems from OSQP
and there appear to be problems building more recent versions of ECOS on M1 machines.
See `this comment <https://github.com/cvxpy/cvxpy/issues/1190#issuecomment-994613793>`_
on the CVXPY repo and
`this issue <https://github.com/embotech/ecos-python/issues/33>`_ on the ECOS repo
for more information.


Running the test suite
------------------------------------
CVXPY comes with an extensive test suite, which can be run after installing `pytest`_.
If installed from source, navigate to the root of the repository and run

  ::

      pytest

To run the tests when CVXPY was not installed from source, use

  ::

      pytest --pyargs cvxpy.tests

Install with CVXOPT and GLPK support
------------------------------------

CVXPY supports the `CVXOPT`_ solver.
Additionally, through CVXOPT, CVXPY supports the `GLPK`_ solver. On `most
platforms <https://cvxopt.org/install/index.html#installing-a-pre-built-package>`_,
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
See the `GUROBI <https://www.gurobi.com/>`_ website for installation instructions.

Install with MOSEK support
---------------------------

CVXPY supports the MOSEK solver.
Simply install MOSEK such that you can ``import mosek`` in Python.
See the `MOSEK <https://www.mosek.com/>`_ website for installation instructions.

Install with XPRESS support
---------------------------

CVXPY supports the FICO Xpress solver.
Simply install XPRESS such that you can ``import xpress`` in Python.
See the `Xpress Python documentation <https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html>`_ pages for installation instructions.

Install with Cbc (Clp, Cgl) support
-----------------------------------
CVXPY supports the `Cbc <https://github.com/coin-or/Cbc>`_ solver (which includes Clp and Cgl) with the help of `cylp <https://github.com/coin-or/CyLP>`_.
Simply install cylp and the corresponding prerequisites according to the `instructions <https://github.com/coin-or/CyLP#cylp>`_, such you can import this library in Python.

Install with CPLEX support
--------------------------

CVXPY supports the CPLEX solver.
Simply install CPLEX such that you can ``import cplex`` in Python.
See the `CPLEX <https://www.ibm.com/support/knowledgecenter/SSSA5P>`_ website for installation instructions.

Install with SDPT3 support
--------------------------

The `sdpt3glue package <https://github.com/TrishGillett/pysdpt3glue>`_ allows you to model problems with CVXPY and solve them with SDPT3.

Install with NAG support
------------------------

CVXPY supports the NAG solver.
Simply install NAG such that you can ``import naginterfaces`` in Python.
See the `NAG <https://www.nag.co.uk/nag-library-python>`_ website for installation instructions.

Install with GLOP and PDLP support
----------------------------------

CVXPY supports the GLOP and PDLP solvers. Both solvers are provided by
the open source `OR-Tools <https://github.com/google/or-tools>`_ package.
Install OR-Tools such that you can run ``import ortools`` in Python. OR-Tools
version 9.3 or greater is required.

Install with SCIP support
-------------------------

CVXPY supports the SCIP solver through the ``pyscipopt`` Python package;
we do not support pyscipopt version 4.0.0 or higher; you need to use pyscipopt version 3.x.y
for some (x,y).
See the `PySCIPOpt <https://github.com/SCIP-Interfaces/PySCIPOpt#installation>`_ github for installation instructions.

CVXPY's SCIP interface does not reliably recover dual variables for constraints. If you require dual variables for a continuous problem, you will need to use another solver. We welcome additional contributions to the SCIP interface, to recover dual variables for constraints in continuous problems.

Install with SCIPY support
-------------------------

CVXPY supports the SCIPY solver for LPs.
This requires the `SciPy`_ package in Python which should already be installed as it is a requirement for CVXPY. `SciPy`_'s "interior-point" and "revised-simplex" implementations are written in python and are always available however the main advantage of this solver, is its ability to use the `HiGHS`_ LP solvers (which are written in C++) that comes bundled with `SciPy`_ version 1.6.1 and higher.

Install without default solvers
-------------------------

CVXPY can also be installed without the default solver dependencies.
This can be useful if the intention is to only use non-default solvers.

The solver-less installation, ``cvxpy-base``, can currently be installed through pip and conda.

Installing using pip:

  ::

      pip install cvxpy-base


Installing using conda:

  ::

      conda install cvxpy-base


.. _conda: https://docs.conda.io/en/latest/
.. _CVXOPT: https://cvxopt.org/
.. _OSQP: https://osqp.org/
.. _ECOS: https://github.com/ifa-ethz/ecos
.. _SCS: https://github.com/cvxgrp/scs
.. _NumPy: https://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _pytest: https://docs.pytest.org/en/latest/
.. _CVXPY git repository: https://github.com/cvxpy/cvxpy
.. _pip: https://pip.pypa.io/
.. _GLPK: https://www.gnu.org/software/glpk/
.. _HiGHS: https://www.maths.ed.ac.uk/hall/HiGHS/#guide
