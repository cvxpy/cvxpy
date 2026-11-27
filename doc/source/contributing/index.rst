.. _contributing:

Contributing
===============

We welcome all contributors to CVXPY. You don't need to be an expert in convex
optimization to help out. Here are simple ways to start contributing immediately:

* Help answer questions on the `CVXPY discord <https://discord.gg/4urRQeGBCr>`_,
  `Github discussions <https://github.com/cvxpy/cvxpy/discussions>`_,
  or `StackOverflow <https://stackoverflow.com/questions/tagged/cvxpy>`_

* Read the CVXPY source code and improve the documentation, or address TODOs

* Fix typos or otherwise enhance the `website documentation <https://github.com/cvxpy/cvxpy/tree/master/doc>`_

* Browse the `issue tracker <https://github.com/cvxpy/cvxpy/issues>`_, and work on unassigned bugs or feature requests

* Polish the `example library <https://www.cvxpy.org/examples/index.html>`_

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first by opening a GitHub issue to make sure that your
priorities align with ours.
We've identified some specific development opportunities in the next section.

CVXPY contributors must follow `our code of conduct <https://github.com/cvxpy/cvxpy/blob/master/CODE_OF_CONDUCT.md>`_.
Overall development is guided by `our governance structure <https://github.com/cvxpy/org/blob/main/governance.md>`_.

The rest of this page goes into more detail on how to contribute to CVXPY.

.. _contrib_roadmap:

Wishlist
--------

Here is a non-exhaustive list of opportunities to make prominent contributions to CVXPY.

We've roughly categorized the opportunities as whether they amount to small-, medium-, or large-scope projects.
New contributors are encouraged to focus on small or medium projects.

If you're interested in working on a large project, join the `Discord <https://discord.gg/4urRQeGBCr>`_ and let us know!
We'll be happy to provide guidance and advice.

Small scope projects
 - `State required cone types for atoms <https://github.com/cvxpy/cvxpy/issues/574>`_.
 - Have MOSEK interface choose between solving the primal and the dual [`2107 <https://github.com/cvxpy/cvxpy/issues/2107>`_, `1403 <https://github.com/cvxpy/cvxpy/issues/1403>`_].
 - `More verbose logging info (expression tree nodes, DCP time) <https://github.com/cvxpy/cvxpy/issues/2674>`_.
 - Improve type checking for compatibility with Pyright.
 - Add explanation and examples of explicit conic constraints to `Advanced constraints <https://www.cvxpy.org/tutorial/constraints/index.html>`_.

Medium scope projects
 - `Post-solver feasibility checks <https://github.com/cvxpy/cvxpy/issues/434>`_.
 - `Vectorize the quad_over_lin atom <https://github.com/cvxpy/cvxpy/issues/1197>`_.
 - Caching computation of lin_ops during matrix stuffing.
 - Introduce sparse variables in constraint formulation.
 - N-dimensional matmul with broadcasting supported.
 - `Marimo examples (maybe using N-d arrays) <https://github.com/cvxpy/cvxpy/issues/2618>`_.
 - `QP support for PDLP solver <https://github.com/cvxpy/cvxpy/issues/2868>`_.
 - `More array manipulation atoms <https://github.com/cvxpy/cvxpy/issues/2567>`_.
 - `Implement cp.block for full numpy.block parity <https://github.com/cvxpy/cvxpy/issues/3469>`_.
 - Support for a set and indexing API similar to `AMPL <https://ampl.com/wp-content/uploads/Chapter-5-Simple-Sets-and-Indexing-AMPL-Book.pdf>`_.

Large scope projects
 - Problem serialization [`532 <https://github.com/cvxpy/cvxpy/issues/532>`_, `1333 <https://github.com/cvxpy/cvxpy/issues/1333>`_, `1438 <https://github.com/cvxpy/cvxpy/issues/1438>`_].
 - Batch support for parameters.
 - `Numerical debugging/analysis tool <https://github.com/cvxpy/cvxpy/issues/3022>`_.
 - Infeasibility debugging tool (IIS or slack minimization).
 - Full parity with NumPy for ND array behavior.
 - Support for GSCOPT [`2889 <https://github.com/cvxpy/cvxpy/discussions/2889>`_].
 - Disable constraints using parameters [`3041 <https://github.com/cvxpy/cvxpy/discussions/3041>`_]
 - Explicit variable/dual recovery maps in reductions. Currently, reductions use ad-hoc logic
   to map between reduced and original variables/duals during solution recovery. Reductions
   should instead return explicit forward and inverse maps for primal and dual variables.

General principles
------------------


Development environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Start by forking the CVXPY repository and installing CVXPY
:ref:`from source <install_from_source>`.
You should configure git on your local machine before changing any code.
Here's one way CVXPY contributors might configure git:

1. Tell git about the existence of the official CVXPY repo:
::

    git remote add upstream https://github.com/cvxpy/cvxpy.git

2. Fetch a copy of the official master branch:
::

    git fetch upstream master

3. Create a local branch which will track the official master branch:
::

    git branch --track official_master upstream/master

The *only* command you should use on the ``official_master`` branch is ``git pull``.
The purpose of this tracking branch is to allow you to easily sync with the main
CVXPY repository. Such an ability can be a huge help in resolving any merge conflicts
encountered in a pull request. For simple contributions, you might never use this branch.

4. Switch back to your forked master branch:
::

    git checkout master

5. Resume work as usual!

Contribution checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

Contributions are made through
`pull requests <https://help.github.com/articles/using-pull-requests/>`_.
Before sending a pull request, make sure you do the following:

- Add our :ref:`license <contrib_license>` to new files

- Check that your code adheres to our :ref:`coding style <contrib_style>`.

- :ref:`Write<contrib_unittests>` unittests.

- :ref:`Run<contrib_run_tests>` the unittests and check that they're passing.

- :ref:`Run the benchmarks<contrib_run_benchmarks>` to make sure your change doesn't introduce a regression

Once you've made your pull request, a member of the CVXPY development team
will assign themselves to review it. You might have a few back-and-forths
with your reviewer before it is accepted, which is completely normal. Your
pull request will trigger continuous integration tests for many different
Python versions and different platforms. If these tests start failing, please
fix your code and send another commit, which will re-trigger the tests.


.. _contrib_license:

License
~~~~~~~~~~~~~~~~~~~~~~~~~
Please add the following license to new files:

::

  """
  Copyright, the CVXPY authors

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  """

.. _contrib_style:

Code style
~~~~~~~~~~~~~~~~~~~~~~~~~
We use `ruff <https://beta.ruff.rs/docs/>`_ to enforce our Python coding
style. Before sending us a pull request, navigate to the project root
and run

::

    pip install ruff
    ruff check cvxpy

to make sure that your changes abide by our style conventions. Please fix any
errors that are reported before sending the pull request.

Optionally, the package `pre-commit <https://pre-commit.com/>`_ can be installed
to check these conventions automatically before every commit.

::

     pip install pre-commit
     pre-commit install

.. _contrib_unittests:

Writing unit tests
~~~~~~~~~~~~~~~~~~~~~~~~~
Most code changes will require new unit tests. (Even bug fixes require unit tests,
since the presence of bugs usually indicates insufficient tests.) CVXPY tests
live in the directory `cvxpy/tests`, which contains many files, each of which
contains many unit tests. When adding tests, try to find a file in which your
tests should belong; if you're testing a new feature, you might want to create
a new test file.

We use the standard Python `unittest <https://docs.python.org/3/library/unittest.html>`_
framework for our tests. Tests are organized into classes, which inherit from
``BaseTest`` (see ``cvxpy/tests/base_test.py``). Every method beginning with ``test_`` is a unit
test.

.. _contrib_run_tests:

Running unit tests
~~~~~~~~~~~~~~~~~~~~~~~~~
We use ``pytest`` to run our unit tests, which you can install with ``pip install pytest``.
To run all unit tests, ``cd`` into ``cvxpy/tests`` and run the following command:

::

    pytest

To run tests in a specific file (e.g., ``test_dgp.py``), use

::

    pytest test_dgp.py

To run a specific test method (e.g., ``TestDgp.test_product``), use

::

    pytest test_dgp.py::TestDgp::test_product

Please make sure that your change doesn't cause any of the unit tests to fail.

``pytest`` suppresses stdout by default. To see stdout, pass the ``-s`` flag
to ``pytest``.

.. _contrib_run_benchmarks:

Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~
CVXPY has performance `benchmarks <https://github.com/cvxpy/benchmarks>`_ which test the time to canonicalize problems.
These problems are automatically benchmarked on every commit using `airspeed velocity <https://asv.readthedocs.io>`_ and github actions.

Optionally, the package `act <https://github.com/nektos/act>`_ can be installed
to run github actions locally.

.. _contrib_atoms:

Adding an atom
----------------------

An atom is a function that CVXPY can reason about. It knows the shape, sign,
curvature and monotonicity of the function, so it can apply the DCP rules.
It also knows how to turn the function into constraints that a solver understands.
This section goes through the pieces of a new atom in the order you will usually
write them. Please follow the :ref:`contribution checklist <contrib_license>` above
as for any other change.

Open a GitHub issue before you start, so that the CVXPY developers can tell you
whether the atom fits and whether something similar exists already.
If the function can be written with existing atoms, a short function that returns
an expression is often better than a new atom.
``cvxpy.huber`` and ``cvxpy.trace`` are functions that choose which atom to build.

The best way to learn the structure is to read a similar atom from start to end.
Good ones to start from are ``xexp`` (elementwise, exponential cone),
``huber`` (elementwise, second order cone), ``quad_over_lin`` (second order cone)
and ``trace`` (affine). The example below is a small made-up atom.
File names are relative to the root of the CVXPY repository.

Step 1: write the atom class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Atoms live in ``cvxpy/atoms``. Elementwise atoms go in ``cvxpy/atoms/elementwise``
and affine atoms go in ``cvxpy/atoms/affine``. Every atom derives from ``Atom``
in ``cvxpy/atoms/atom.py``, usually through one of its subclasses:

- ``AffAtom`` for affine functions (``cvxpy/atoms/affine/affine_atom.py``),
- ``Elementwise`` for functions applied to each entry (``cvxpy/atoms/elementwise/elementwise.py``),
- ``AxisAtom`` for reductions over an ``axis`` such as ``max`` or ``norm1`` (``cvxpy/atoms/axis_atom.py``).

The class describes the function to the DCP rules. These methods are needed:

- ``numeric(values)`` evaluates the function on a list of numeric argument values.
- ``shape_from_args()`` returns the shape of the output.
- ``sign_from_args()`` returns ``(is_nonneg, is_nonpos)``. It may look at the arguments.
- ``is_atom_convex()`` and ``is_atom_concave()`` give the curvature of the function
  itself, as if its arguments were affine.
- ``is_incr(idx)`` and ``is_decr(idx)`` say whether the function is nondecreasing or
  nonincreasing in argument ``idx``. They may depend on the sign of the argument.
  For example ``huber`` is nondecreasing in ``x`` when ``x`` is known to be nonnegative.
- ``_grad(values)`` returns the gradient with respect to each argument as a sparse matrix,
  or ``None`` where the function is not differentiable.

These are optional:

- ``validate_arguments()`` raises ``ValueError`` for arguments that are not allowed.
- ``_domain()`` returns constraints that describe where the function is defined.
- ``get_data()`` returns constructor arguments that are not expressions.
  CVXPY rebuilds an atom as ``type(atom)(*args, *data)``, so the constructor
  and ``get_data()`` must agree. ``huber`` keeps its parameter ``M`` this way.
- ``is_atom_log_log_convex()`` and ``is_atom_log_log_concave()`` for use in
  :ref:`DGP <dgp>` problems.

Here is an example. It is the Euclidean norm of a vector with an extra entry equal
to one, :math:`\sqrt{1 + \|x\|_2^2}`. The function is convex and positive.
The file would be ``cvxpy/atoms/aug_norm.py``.

.. code:: python

    import numpy as np
    import scipy.sparse as sp

    from cvxpy.atoms.atom import Atom


    class aug_norm(Atom):
        """The Euclidean norm of x with a 1 appended, sqrt(1 + ||x||^2)."""

        def __init__(self, x) -> None:
            super(aug_norm, self).__init__(x)

        def numeric(self, values):
            return np.sqrt(1 + np.sum(np.square(values[0])))

        def shape_from_args(self) -> tuple[int, ...]:
            return ()

        def sign_from_args(self) -> tuple[bool, bool]:
            return (True, False)

        def is_atom_convex(self) -> bool:
            return True

        def is_atom_concave(self) -> bool:
            return False

        def is_incr(self, idx) -> bool:
            return self.args[idx].is_nonneg()

        def is_decr(self, idx) -> bool:
            return self.args[idx].is_nonpos()

        def _grad(self, values):
            x = values[0]
            grad = x / np.sqrt(1 + np.sum(np.square(x)))
            return [sp.csc_matrix(grad.reshape((-1, 1)))]

The gradient of a scalar atom is a column with one row per entry of the argument.
For an atom with array output it is a matrix from the vectorized argument to the
vectorized output. ``Elementwise.elemwise_grad_to_diag`` builds it for elementwise atoms.

Step 2: write a canonicalizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solvers only understand affine expressions and constraints on cones.
The ``Dcp2Cone`` reduction (``cvxpy/reductions/dcp2cone/dcp2cone.py``) replaces every
atom in a problem by an affine expression plus some constraints.
What you write for this depends on the atom.

An affine atom implements ``graph_implementation`` on its class. It builds a
linear operator from the linear operators of the arguments.
``Trace`` in ``cvxpy/atoms/affine/trace.py`` is a short example.
Most new atoms are not affine, so you can usually skip this.

Any other atom gets a canonicalizer. This is a function ``canon(expr, args, solver_context=None)``
where ``expr`` is the atom and ``args`` are its arguments after they were canonicalized,
so they are affine. The function returns an affine expression that stands in for the atom,
and a list of constraints that make the expression equal to the atom at the optimum.
It creates new ``Variable`` objects and constrains them with the cones in ``cvxpy.constraints``.
For our example the constraint is one second order cone.
The file would be ``cvxpy/reductions/dcp2cone/canonicalizers/aug_norm_canon.py``.

.. code:: python

    from cvxpy.atoms.affine.hstack import hstack
    from cvxpy.atoms.affine.vec import vec
    from cvxpy.constraints.second_order import SOC
    from cvxpy.expressions.variable import Variable
    from cvxpy.utilities.solver_context import SolverInfo


    def aug_norm_canon(expr, args, solver_context: SolverInfo | None = None):
        t = Variable()
        x = vec(args[0], order="F")
        return t, [SOC(t, hstack([1, x]))]

The other files in that folder show how to build exponential cone, power cone and
semidefinite constraints. A canonicalizer can call the canonicalizer of another atom,
as ``xexp_canon.py`` does with ``power_approx_canon``.

Step 3: register the canonicalizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open ``cvxpy/reductions/dcp2cone/canonicalizers/__init__.py``. Import the canonicalizer
and add it to the ``CANON_METHODS`` dictionary:

.. code:: python

    from cvxpy.reductions.dcp2cone.canonicalizers.aug_norm_canon import *

    CANON_METHODS = {
        ...
        aug_norm : aug_norm_canon,
    }

The key must be the class of the atom, not a factory function.
``cvxpy.huber`` is a function, so the key for it is ``HuberAtom``.
``cvxpy/tests/test_canon_methods.py`` checks this for every registry.
An atom that is not in ``CANON_METHODS`` is passed on unchanged, and the solver cannot handle it.

An atom that can appear in the objective of a quadratic program may also need an entry in
``QUAD_CANON_METHODS`` in ``cvxpy/reductions/dcp2cone/canonicalizers/quad/__init__.py``.
Only a few atoms such as ``quad_over_lin`` and ``huber`` have one.
Piecewise linear atoms such as ``abs`` and ``max`` have their canonicalizers in
``cvxpy/reductions/eliminate_pwl/canonicalizers``, and ``CANON_METHODS`` imports them from there.

Step 4: export the atom and list its cones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open ``cvxpy/atoms/__init__.py``. Import the atom there so that it is available as
``cvxpy.aug_norm``:

.. code:: python

    from cvxpy.atoms.aug_norm import aug_norm

The same file has lists of the atoms that need each kind of cone: ``SOC_ATOMS``,
``EXP_ATOMS``, ``PSD_ATOMS``, ``POWCONE_ATOMS``, ``POWCONE_ND_ATOMS`` and ``NONNEG_ATOMS``.
CVXPY reads them in ``cvxpy/problems/problem_form.py`` to find the cones that a problem
needs, and then it picks a solver that supports those cones.
Add your atom to the list for every cone that its canonicalizer uses.
Our example goes in ``SOC_ATOMS``. If an atom is missing from the lists, CVXPY may pick a
solver that cannot handle the constraints that the canonicalizer creates.

Step 5: write tests
~~~~~~~~~~~~~~~~~~~~~~~

Tests live in ``cvxpy/tests``. Most atoms need entries in these files:

- ``test_atoms.py`` checks the shape, sign, curvature and monotonicity, including the
  cases where they depend on the sign of the argument. ``test_xexp`` is a short example.
- ``test_constant_atoms.py`` solves a small problem for each atom, with the arguments given as
  constants, variables and parameters, and compares the result with a known value.
  Add a tuple to ``atoms_minimize`` or ``atoms_maximize`` with the function,
  the output shape, the argument values and the expected result:

  .. code:: python

      (aug_norm, tuple(), [[3, 4]], Constant([math.sqrt(26)])),

- ``test_grad.py`` compares ``_grad`` with finite differences. Add an
  ``AtomTestConfig`` for the atom.
- ``test_dgp2dcp.py`` is for atoms that are log-log convex or concave.

:ref:`Run <contrib_run_tests>` the tests in the files you changed, and then the full test suite.

Step 6: document the atom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give the class a docstring that says what the function is and lists its arguments,
in the style of the other atoms. Then add the atom to the docs:

- ``doc/source/api_reference/``: an entry in ``cvxpy.atoms.elementwise.rst``,
  ``cvxpy.atoms.affine.rst`` or ``cvxpy.atoms.other_atoms.rst``. Copy the ``.. _name:``
  label and the ``autoclass`` or ``autofunction`` directive from a neighbouring entry.
- ``doc/source/functions/functions_table.rst``: a row with the meaning, domain,
  DCP properties and curvature.
- ``doc/source/tutorial/functions/index.rst``: the same row in the table for
  scalar, elementwise or vector and matrix functions.
- ``doc/source/tutorial/dgp/index.rst``: a row if the atom is log-log convex or concave.

Install ``doc/requirements.txt`` and run ``make html`` in the ``doc`` folder to check that
the new entries render and that the math in the docstring displays correctly.

Other kinds of problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An atom that should work in :ref:`DGP <dgp>` problems needs a rule in
``cvxpy/reductions/dgp2dcp/canonicalizers``. An atom for the nonlinear solvers (DNLP)
sets ``is_atom_smooth()`` and may need a rule in ``cvxpy/reductions/dnlp2smooth/canonicalizers``.
Ask the developers in your issue whether the atom should support either.

.. _contrib_solver:

Solver interfaces
----------------------

Third-party numerical optimization solvers are the lifeblood of CVXPY.
We are very grateful to anyone who would be willing to volunteer their time to
improve our existing solver interfaces, or create interfaces to new solvers.
Improving an existing interface can usually be handled like fixing a bug.
Creating a new interface requires much more work, and warrants coordination
with CVXPY principal developers before writing any code.

This section of the contributing guide outlines considerations when adding new solver interfaces.
For the time being, we only have documentation for conic solver interfaces.
Additional documentation for QP solver interfaces is forthcoming.

.. warning::

    This documentation is far from complete! It only tries to cover the absolutely
    essential parts of writing a solver interface. It also might not do that in
    a spectacular way -- we welcome all feedback on this part of the documentation.

.. warning::

    The developers try to keep this documentation up to date, however at any given time
    it might contain inaccurate information! It's very important that you contact the
    CVXPY developers before writing a solver interface, if for no other reason than to
    prompt us to double-check the accuracy of this guide.

Conic solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conic solvers require that the objective is a linear function of the
optimization variable; constraints must be expressed using convex cones and
affine functions of the optimization variable.
The codepath for conic solvers begins with
`reductions/solvers/conic_solvers <https://github.com/cvxpy/cvxpy/tree/master/cvxpy/reductions/solvers/conic_solvers>`_
and in particular with the class ``ConicSolver`` in
`conic_solver.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/conic_solver.py>`_.

Let's say you're writing a CVXPY interface for the "*Awesome*" conic solver,
and that there's an existing package ``AwesomePy`` for calling *Awesome* from python.
In this case you need to create a file called ``awesome_conif.py`` in the same folder as ``conic_solver.py``.
Within ``awesome_conif.py`` you will define a class ``Awesome(ConicSolver)``.
The ``Awesome(ConicSolver)`` class will manage all interaction between CVXPY and the
existing ``AwesomePy`` python package. It will need to implement six functions:
- import_solver,
- name,
- accepts,
- apply,
- solve_via_data, and
- invert.

The first three functions are very easy (often trivial) to write.
The remaining functions are called in order: ``apply`` stages data for ``solve_via_data``,
``solve_via_data`` calls the *Awesome* solver by way of the existing third-party
``AwesomePy`` package, and ``invert`` transforms the output from ``AwesomePy`` into
the format that CVXPY expects.

Key goals in this process are that the output of ``apply`` should be as close as possible
to the *Awesome*'s standard form, and that ``solve_via_data`` should be kept short.
The complexity of ``Awesome(ConicSolver).solve_via_data`` will depend on ``AwesomePy``.
If ``AwesomePy`` allows very low level input-- passed by one or two matrices,
and a handful of numeric vectors --then you'll be in a situation like ECOS or GLPK.
If the ``AwesomePy`` package requires that you build an object-oriented model,
then you're looking at something closer to the MOSEK, GUROBI, or NAG interfaces.
Writing the ``invert`` function may require nontrivial effort to properly recover dual variables.

CVXPY's conic form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CVXPY converts an optimization problem to an explicit form at the last possible moment.
When CVXPY presents a problem in a concrete form, it's over a single vectorized
optimization variable, and a flattened representation of the feasible set.
The abstraction for the standard form is

.. math::

   (P) \quad \min\{ c^T x + d \,:\, x \in \mathbb{R}^{n},\, A x + b \in K \}

where :math:`K` is a product of elementary convex cones. The design of CVXPY allows
for any cone supported by a target solver, but the current elementary convex cones are

1. The zero cone :math:`y = 0 \in \mathbb{R}^m`.
2. The nonnegative cone :math:`y \geq 0 \in \mathbb{R}^m`.
3. The second order cone

.. math::

    (u,v) \in K_{\mathrm{soc}}^n \doteq \{ (t,x) \,:\, t \geq \|x\|_2  \} \subset \mathbb{R} \times \mathbb{R}^n.

4. One of several vectorized versions of the positive semidefinite cone.
5. The exponential cone

.. math::

    (u,v,w) \in K_e \doteq \mathrm{cl}\{(x,y,z) |  z \geq y \exp(x/y), y>0\}.

6. The 3-dimensional power cone, parameterized by a number :math:`\alpha\in (0, 1)`:

.. math::

    (u,v) \in K_{\mathrm{pow}}^{\alpha} \doteq \{ (x,y,z) \,:\, x^{\alpha}y^{1-\alpha} \geq |z|, (x,y) \geq 0 \}.

We address the vectorization options for the semidefinite cones later.
For now it's useful to say that the ``Awesome(ConicSolver)`` class will access an
explicit representation for problem :math:`(P)` in ``apply``, with a code snippet like

.. code::

    # from cvxpy.constraints import Zero, NonNeg, SOC, PSD, ExpCone, PowCone3D
    #  ...
    constr_map = problem.constr_map
    cone_dims = problem.cone_dims
    c, d, A, b = problem.apply_parameters()

``apply`` receives a program whose constraint rows are already in the layout
your ``EXP_CONE_ORDER`` asks for: the ``ConeFormat`` reduction does that once,
immediately before the solver, so an interface must not re-derive it.

The variable ``constr_map`` is a dict of lists of CVXPY Constraint objects.
The dict is keyed by the references to CVXPY's Zero, NonNeg, SOC, PSD, ExpCone,
and PowCone3D classes. You will need to interact with these constraint classes during
dual variable recovery.
For the other variables in that code snippet ...

- ``c, d`` define the objective function ``c @ x + d``, and
- ``A, b, cone_dims`` define the abstractions :math:`A`, :math:`b`, :math:`K` in problem  :math:`(P)`.

The first step in writing a solver interface is to understand the exact
meanings of ``A, b, cone_dims``, so that you can correctly build a primal
problem using the third-party ``AwesomePy`` interface to the *Awesome* solver.
The ``cone_dims`` object is an instance of the ConeDims class, as defined in
`cone_matrix_stuffing.py
<https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py>`_;
``A`` is a SciPy sparse matrix, and ``b`` is a numpy ndarray with ``b.ndim == 1``.
The rows of ``A`` and entries of ``b`` are given in a very specific order, as described below.

- Equality constraints are found in the first ``cone_dims.zero`` rows of ``A`` and entries of ``b``.
  Letting ``eq = cone_dims.zero``, the constraint is

  .. code::

      A[:eq, :] @ x + b[:eq] == 0.

- Inequality constraints occur immediately after the equations.
  If for example ``ineq = cone_dims.nonneg`` then the feasible
  set has the constraint

  .. code::

      A[eq:eq + ineq, :] @ x + b[eq:eq + ineq] >= 0.

- Second order cone (SOC) constraints are handled after inequalities.
  Here, ``cone_dims.soc`` is a *list of integers* rather than a single integer.
  Supposing ``cone_dims.soc[0] == 10``, the first second order cone constraint appearing
  in this optimization problem would involve 10 rows of ``A`` and 10 entries of ``b``.
  The SOC vectorization we use is given by :math:`K_{\mathrm{soc}}^n` as defined above.
- PSD constraints follow SOC constraints.
  For most solver interfaces it is a good idea to make a deliberate decision about how to
  handle the vectorization, which amounts to implementing ``Awesome(ConicSolver).psd_format_mat``.
  If you do nothing, then the vectorization will behave as in ``ConicSolver.psd_format_mat``,
  which takes a PSD constraint of order :math:`n` and maps it to :math:`n^2` rows of :math:`A` and
  entries of :math:`b`.
  You can also borrow from ``SCS.psd_format_mat`` which maps an order :math:`n` PSD constraint
  to :math:`n(n+1)/2` suitably scaled rows of :math:`A` and entries of :math:`b`, or
  ``MOSEK.psd_format_mat`` which behaves identically to SCS except for the scaling.
- The next block of ``3 * cone_dims.exp`` rows in ``A, b`` correspond to consecutive
  three-dimensional exponential cones, as defined by :math:`K_e` above.
- The final block of ``3 * len(cone_dims.p3d)`` rows in ``A, b`` correspond to
  three-dimensional power cones defined by :math:`K_{\mathrm{pow}}^{\alpha}`, where the
  i-th triple of rows has ``alpha = cone_dims.p3d[i]``.

If *Awesome* supports nonlinear constraints like SOC, ExpCone, PSD, or PowCone3D, then
it's possible that you will need to transform data ``A, b`` in order to write these constraints in
the form expected by ``AwesomePy``.
The most common situations are when ``AwesomePy`` parametrizes the second-order cone
as :math:`K = \{ (x,t) \,:\, \|x\|\leq t \} \subset \mathbb{R}^n \times \mathbb{R}`,
or when it parametrizes :math:`K_e \subset \mathbb{R}^3` as some permutation of
what we defined earlier.

An alternative conic form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some conic solvers do not natively support problem formats like (P) described in
the previous section. Instead, the solver requires problem statements like

.. math::

   (Dir) \quad \min\{ f^T z  \,:\, z \in K,\, G z = h \}.

Problem (Dir) uses so-called "direct" conic constraints :math:`z \in K`. If you are
writing an interface for a solver which works this way, you should use the
``Dualize`` reduction on the standard CVXPY problem data given in (P).
Using the Dualize reduction will avoid introduction unnecessary slack variables
for continuous problems, but it is not applicable for problems with integer constraints.
Therefore if your solver supports integer constraints, make sure to also use the
``Slacks`` reduction for that code path.

The MOSEK interface uses both of the reductions mentioned above.

Dual variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual variable extraction should be handled in ``Awesome(ConicSolver).invert``.
To perform this step correctly, it's necessary to consider how CVXPY forms
a Lagrangian for the primal problem :math:`(P)`.
Let's say that the affine map :math:`Ax + b` in the feasible set
:math:`Ax + b \in K \subset \mathbb{R}^m` is broken up into six blocks of sizes
:math:`m_1,\ldots,m_6` where the blocks correspond (in order) to zero-cone, nonnegative cone,
second-order cone, vectorized PSD cone, exponential cone, and 3D power cone constraints.
Then CVXPY defines the dual to :math:`(P)` by forming a Lagrangian

.. math::

    \mathcal{L}(x,\mu_1,\ldots,\mu_6) = c^T x - \sum_{i=i}^6 \mu_i^T (A_i x + b_i)

in dual variables :math:`\mu_1 \in \mathbb{R}^{m_1}`, :math:`\mu_2 \in \mathbb{R}^{m_2}_+`,
and :math:`\mu_i \in K_i^* \subset \mathbb{R}^{m_i}` for :math:`i \in \{3,4,5,6\}`.
Here, :math:`K_i^*` denotes the dual cone to :math:`K_i` under the standard inner product.

More remarks on dual variables (particularly SOC dual variables) can be found in
`this comment on a GitHub thread <https://github.com/cvxpy/cvxpy/issues/948#issuecomment-592781675>`_.

Most concrete implementations of the ConicSolver class use a common set of helper
functions for dual variable recovery, found in
`reductions/solvers/utilities.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/utilities.py>`_.


Registering a solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Correctly implementing ``Awesome(ConicSolver)`` isn't enough to call *Awesome* from CVXPY.
You need to make edits in a handful of other places, namely

- `conic_solvers/__init__.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/__init__.py>`_,
- `solvers/defines.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/reductions/solvers/defines.py>`_, and
- `cvxpy/__init__.py <https://github.com/cvxpy/cvxpy/blob/master/cvxpy/__init__.py>`_.

The existing content of those files should make it clear what's needed
to add *Awesome* to CVXPY.

Writing tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for  ``Awesome(ConicSolver)`` should be placed in `cvxpy/tests/test_conic_solvers.py
<https://github.com/cvxpy/cvxpy/blob/master/cvxpy/tests/test_conic_solvers.py>`_.
The overwhelming majority of tests in that file only take a single line, because
we make consistent use of a general testing framework defined in
`solver_test_helpers.py
<https://github.com/cvxpy/cvxpy/blob/master/cvxpy/tests/solver_test_helpers.py>`_.
Here are examples of helper functions we invoke in ``test_conic_solvers.py``,

.. code::

    class StandardTestSDPs(object):

        @staticmethod
        def test_sdp_1min(solver, places=4, **kwargs):
            sth = sdp_1('min')
            sth.solve(solver, **kwargs)
            sth.verify_objective(places=2)  # only 2 digits recorded.
            sth.check_primal_feasibility(places)
            sth.check_complementarity(places)
            sth.check_dual_domains(places)  # check dual variables are PSD.

    ...

    class StandardTestSOCPs(object):

        @staticmethod
        def test_socp_0(solver, places=4, **kwargs):
            sth = socp_0()
            sth.solve(solver, **kwargs)
            sth.verify_objective(places)
            sth.verify_primal_values(places)
            sth.check_complementarity(places)

    ...

        @staticmethod
        def test_mi_socp_1(solver, places=4, **kwargs):
            sth = mi_socp_1()
            sth.solve(solver, **kwargs)
            # mixed integer problems don't have dual variables,
            #   so we only check the optimal objective and primal variables.
            sth.verify_objective(places)
            sth.verify_primal_values(places)

Notice the comments in the predefined functions.
In ``test_sdp_1min``, we override a user-supplied value for ``places`` with
``places=2`` when checking the optimal objective function value.
We also go through extra effort to check that the dual variables are PSD
matrices.
In ``test_mi_socp_1`` we're working with a mixed-integer problem, so
there are no dual variables at all.
You should use these predefined functions partly because they automatically check
what's most appropriate for the problem at hand.

Each of these predefined functions first constructs a SolverTestHelper object ``sth``
which contains appropriate test data. The ``.solve`` function for the
SolverTestHelper class is a simple wrapper around ``prob.solve`` where
``prob`` is a CVXPY Problem. In particular, any keyword arguments
passed to ``sth.solve`` will be passed to ``prob.solve``. This allows you to
call modified versions of a test with different solver parameters, for example

.. code::

    def test_mosek_lp_1(self):
        # default settings
        StandardTestLPs.test_lp_1(solver='MOSEK')  # 4 places
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='MOSEK', places=6, bfs=True)



.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _CVXOPT: https://cvxopt.org/
.. _NumPy: https://www.numpy.org/
.. _SciPy: https://www.scipy.org/
