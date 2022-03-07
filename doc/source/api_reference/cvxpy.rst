API Documentation
=================
CVXPY is designed to be intuitive enough so that it may be used
without consulting an API reference; the tutorials will
suffice in acquainting you with our software. Nonetheless, we include here an
API reference for those who are comfortable reading technical documentation.

All of the documented classes and functions are imported into the
``cvxpy`` namespace; this means that they can be used by simply writing
``cvxpy.symbol``, where ``symbol`` is the name of your class or function of
choice, so long as you import the ``cvxpy`` package in your python source file.

The documentation is grouped five sections: *atoms*, *constraints*,
*expressions*, *problems*, and *reductions*. The atoms section
documents the classes implementing atomic mathematical functions, like
``exp``, ``log``, and ``sqrt``; the constraints section documents
the constraints that can be imposed upon variables; the expressions section
documents the classes implementing mathematical expression trees, including the
:class:`~cvxpy.expressions.variable.Variable` and
:class:`~cvxpy.expressions.constants.parameter.Parameter` classes; the
problem section documents the
:class:`~cvxpy.problems.problem.Problem` class and other related classes;
the reductions section documents principled operations that convert
problems from a particular form to another equivalent form; and
the transforms section documents additional operations available for manipulating
CVXPY objects;


.. toctree::
   :maxdepth: 1

   Atoms <cvxpy.atoms>
   Constraints <cvxpy.constraints>
   Expressions <cvxpy.expressions>
   Problems <cvxpy.problems>
   Transforms <cvxpy.transforms>
   Reductions <cvxpy.reductions>
