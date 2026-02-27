.. _cvxpygen:

CVXPYgen: Code generation with CVXPY
=====================================

CVXPYgen takes a convex optimization problem family modeled with CVXPY and generates a custom solver implementation in C.
This generated solver is specific to the problem family and accepts different parameter values.
In particular, this solver is suitable for deployment on embedded systems.
In addition, CVXPYgen creates a Python wrapper for prototyping and desktop (non-embedded) applications.

An overview of CVXPYgen can be found in our `slides and manuscript <https://web.stanford.edu/~boyd/papers/cvxpygen.html>`_.

CVXPYgen accepts CVXPY problems that are compliant with `Disciplined Convex Programming (DCP) </tutorial/dcp/index.html>`_.
DCP is a system for constructing mathematical expressions with known curvature from a given library of base functions. 
CVXPY uses DCP to ensure that the specified optimization problems are convex.
In addition, problems need to be modeled according to `Disciplined Parametrized Programming (DPP) </tutorial/advanced/index.html#disciplined-parametrized-programming>`_.
Solving a DPP-compliant problem repeatedly for different values of the parameters can be much faster than repeatedly solving a new problem.

For now, CVXPYgen is a separate module, until it will be integrated into CVXPY.
As of today, CVXPYgen works with linear, quadratic, and second-order cone programs.
It also supports differentiating through quadratic programs and computing an
explicit solution to linear and quadratic programs.

This package has similar functionality as the package `cvxpy_codegen <https://github.com/moehle/cvxpy_codegen>`_,
which appears to be unsupported.

.. toctree::
    :maxdepth: 2

    installation
    example
    gradient
    explicit
    tests
