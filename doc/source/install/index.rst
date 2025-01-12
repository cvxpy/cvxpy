.. _install:

Install
=======

CVXPY supports Python 3 on Linux, macOS, and Windows. You can use
pip or conda for installation. You may want to isolate
your installation in a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_,
or a conda environment.

.. card:: Instructions

    .. tab:: pip

        (Windows only) `Download <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_ the Visual Studio build tools for Python 3
        (`instructions <https://docs.google.com/presentation/d/e/2PACX-1vT-p04simYhorAdKstO9F1RK-k6npuyrKWliJ8Wy9uuQoQq_TiFdJA-DK3Kz0irkCEUlmNEH4JScbkwUflXv9c/pub?start=false&loop=false&delayms=3000&resourcekey=0-HEezB2NFstz1GjKDkroJSQ&slide=id.p1>`_).

        (macOS only) Install the Xcode command line tools.

        (optional) Create and activate a virtual environment.

        Install CVXPY using `pip`_:

        ::

            pip install cvxpy

        You can add solver names as "extras"; `pip` will then install the necessary
        additional Python packages.

        ::

            pip install "cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS]"

    .. tab:: conda

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

    .. tab:: Install from source

        We strongly recommend using a fresh virtual environment (virtualenv or conda) when installing CVXPY from source.

        CVXPY has the following dependencies:

        * Python >= 3.9
        * `OSQP`_ >= 0.6.2
        * `CLARABEL`_ >= 0.6.0
        * `SCS`_ >= 3.0
        * `NumPy`_ >= 1.21.6
        * `SciPy`_ >= 1.11.0

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

    .. tab:: Using Codespaces

        We provide support for `GitHub Codespaces <https://github.com/features/codespaces>`_ with
        preconfigured environments for CVXPY development via `devcontainers <https://containers.dev/>`_.
        To get started, click the "Code" button on the CVXPY repository and select "Open with Codespaces".


Install with Additional Solver Support
------------------------------------

.. info:: CVXOPT and GLPK
    :collapsible: open

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

.. info:: GUROBI
    :collapsible:

    CVXPY supports the GUROBI solver.
    Install GUROBI version 7.5.2 or greater such that you can ``import gurobipy`` in Python.
    See the `GUROBI <https://www.gurobi.com/>`_ website for installation instructions.

.. info:: MOSEK
    :collapsible:

    CVXPY supports the MOSEK solver.
    Simply install MOSEK such that you can ``import mosek`` in Python.
    See the `MOSEK <https://www.mosek.com/>`_ website for installation instructions.

.. info:: XPRESS
    :collapsible:

    CVXPY supports the FICO Xpress solver.
    Simply install XPRESS such that you can ``import xpress`` in Python.
    See the `Xpress Python documentation <https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html>`_ pages for installation instructions.

.. info:: Cbc (Clp, Cgl)
    :collapsible:

    CVXPY supports the `Cbc <https://github.com/coin-or/Cbc>`_ solver (which includes Clp and Cgl) with the help of `cylp <https://github.com/coin-or/CyLP>`_.
    Simply install cylp and the corresponding prerequisites according to the `instructions <https://github.com/coin-or/CyLP#cylp>`_, such you can import this library in Python.

.. info:: COPT
    :collapsible:

    CVXPY supports the COPT solver.
    Simply install COPT such that you can ``import coptpy`` in Python.
    See the `COPT <https://github.com/COPT-Public/COPT-Release>`_ release page for installation instructions.

.. info:: CPLEX
    :collapsible:

    CVXPY supports the CPLEX solver.
    Simply install CPLEX such that you can ``import cplex`` in Python.
    See the `CPLEX <https://www.ibm.com/support/knowledgecenter/SSSA5P>`_ website for installation instructions.

.. info:: SDPA
    :collapsible:

    CVXPY supports the SDPA solver.
    Simply install SDPA for Python such that you can ``import sdpap`` in Python.
    See the `SDPA for Python <https://sdpa-python.github.io/docs/installation>`_ website for installation instructions.

.. info:: SDPT3
    :collapsible:

    The `sdpt3glue package <https://github.com/TrishGillett/pysdpt3glue>`_ allows you to model problems with CVXPY and solve them with SDPT3.

.. info:: NAG
    :collapsible:

    CVXPY supports the NAG solver.
    Simply install NAG such that you can ``import naginterfaces`` in Python.
    See the `NAG <https://support.nag.com/numeric/py/nagdoc_latest/readme.html>`_ website for installation instructions.

.. info:: GLOP and PDLP
    :collapsible:

    CVXPY supports the GLOP and PDLP solvers. Both solvers are provided by
    the open source `OR-Tools <https://github.com/google/or-tools>`_ package.
    Install OR-Tools such that you can run ``import ortools`` in Python. OR-Tools
    version 9.3 or greater is required.

.. info:: SCIP
    :collapsible:

    CVXPY supports the SCIP solver through the ``pyscipopt`` Python package.
    See the `PySCIPOpt <https://github.com/SCIP-Interfaces/PySCIPOpt#installation>`_ github for installation instructions.

    CVXPY's SCIP interface does not reliably recover dual variables for constraints. If you require dual variables for a continuous problem, you will need to use another solver. We welcome additional contributions to the SCIP interface, to recover dual variables for constraints in continuous problems.

.. info:: HiGHS
   :collapsible:

   CVXPY supports the HiGHS solver. Run the following command to install the HiGHS python interface.

   .. code-block:: python
    pip install highspy
   
   See the `HiGHS <https://ergo-code.github.io/HiGHS/dev/interfaces/python/>`_ documentation for additional instructions.

.. info:: SCIPY
    :collapsible:

    CVXPY supports the SCIPY solver for LPs and MIPs.
    This requires the `SciPy`_ package in Python, which should already be installed, as it is a requirement for CVXPY.
    `SciPy`_'s "interior-point" and "revised-simplex" implementations are written in Python and are always available.
    However, the main advantage of this solver is its ability to use the `HiGHS`_ LP and MIP solvers (which are written in C++).
    `HiGHS`_ LP solvers come bundled with `SciPy`_ version 1.6.1 and higher.
    The `HiGHS`_ MIP solver comes bundled with version 1.9.0 and higher.

.. info:: PIQP
    :collapsible:

    CVXPY supports the PIQP solver.
    Simply install PIQP such that you can ``import piqp`` in Python.
    See the `PIQP <https://predict-epfl.github.io/piqp/interfaces/python/installation>`_ website for installation instructions.

.. info:: PROXQP
    :collapsible:

    CVXPY supports the PROXQP solver.
    Simply install PROXQP such that you can ``import proxsuite`` in Python.
    See the `proxsuite <https://github.com/simple-robotics/proxsuite#quick-install>`_ github for installation instructions.
    Be aware that PROXQP by default uses dense matrices to represent problem data.
    You may achieve better performance by setting ``backend = 'sparse'`` in your call to ``problem.solve``.

.. info:: QOCO
    :collapsible:

    CVXPY supports the QOCO solver.
    Simply install QOCO such that you can ``import qoco`` in Python.
    See the `QOCO <https://qoco-org.github.io/qoco/install/index.html>`_ website for installation instructions.

.. info:: Without default solvers
    :collapsible:

    CVXPY can also be installed without the default solver dependencies.
    This can be useful if the intention is to only use non-default solvers.

    The solver-less installation, ``cvxpy-base``, can currently be installed through pip and conda.

    Installing using pip:

    .. code-block:: python

        pip install cvxpy-base

    Installing using conda:

    .. code-block:: python

        conda install cvxpy-base

Running the test suite
------------------------------------
CVXPY comes with an extensive test suite, which can be run after installing `pytest`_.
If installed from source, navigate to the root of the repository and run

.. code-block:: python

    pytest

To run the tests when CVXPY was not installed from source, use

.. code-block:: python

    pytest --pyargs cvxpy.tests

.. _conda: https://docs.conda.io/en/latest/
.. _CVXOPT: https://cvxopt.org/
.. _OSQP: https://osqp.org/
.. _ECOS: https://github.com/ifa-ethz/ecos
.. _CLARABEL: https://oxfordcontrol.github.io/ClarabelDocs/
.. _SCS: https://github.com/cvxgrp/scs
.. _NumPy: https://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _pytest: https://docs.pytest.org/en/latest/
.. _CVXPY git repository: https://github.com/cvxpy/cvxpy
.. _pip: https://pip.pypa.io/
.. _GLPK: https://www.gnu.org/software/glpk/
.. _HiGHS: https://highs.dev/
