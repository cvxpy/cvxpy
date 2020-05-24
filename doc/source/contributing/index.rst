.. _contributing:

Contributing
===============

We welcome all contributors to CVXPY. You don't need to be an expert in convex
optimization to help out. Here are simple ways to start contributing immediately:

 * Read the CVXPY source code and improve the documentation, or address TODOs

 * Fix typos or otherwise enhance the `website documentation <https://github.com/cvxgrp/cvxpy/tree/master/doc>`_

 * Browse the `issue tracker <https://github.com/cvxgrp/cvxpy/issues>`_, and work on unassigned bugs or feature requests

 * Polish the `example library <https://github.com/cvxgrp/cvxpy/tree/master/examples>`_

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first by opening a GitHub issue to make sure that your
priorities align with ours.

The remainder of this page goes into more detail on how to contribute to CVXPY.

General principles
----------------------


Development environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Start by forking the CVXPY repository and installing CVXPY
:ref:`from source <install_from_source>`.
You should configure git on your local machine before changing any code.
Here's one way CVXPY contributors might configure git:

 1. Tell git about the existence of the official CVXPY repo:
   ::

    git remote add upstream https://github.com/cvxgrp/cvxpy.git

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
We use `flake8 <http://flake8.pycqa.org/en/latest/>`_ to enforce our Python coding
style. Before sending us a pull request, navigate to the project root
and run

  ::

    flake8 cvxpy/

to make sure that your changes abide by our style conventions. Please fix any
errors that flake8 reports before sending the pull request.

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
We use ``nose`` to run our unit tests, which you can install with ``pip install nose``.
To run all unit tests, ``cd`` into ``cvxpy/tests`` and run the following command:

  ::

    nosetests

To run tests in a specific file (e.g., ``test_dgp.py``), use

  ::

    nosetests test_dgp.py

To run a specific test method (e.g., ``TestDgp.test_product``), use

  ::

    nosetests test_dgp.py:TestDgp.test_product

Please make sure that your change doesn't cause any of the unit tests to fail.

``nosetests`` suppresses stdout by default. To see stdout, pass the ``-s`` flag
to ``nosetests``.

.. _contrib_run_benchmarks:

Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~
CVXPY has a few benchmarks in ``cvxpy/tests/test_benchmarks.py``, which test
the time to canonicalize problems. Please run

  ::

    nosetests -s test_benchmarks.py

with and without your change, to make sure no performance regressions are
introduced. If you are making a code contribution, please include the output of
the above command (with and without your change) in your pull request.

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
`reductions/solvers/conic_solvers <https://github.com/cvxgrp/cvxpy/tree/master/cvxpy/reductions/solvers/conic_solvers>`_
and in particular with the class ``ConicSolver`` in
`conic_solver.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/conic_solver.py>`_.

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

 4. A vectorized version of the positive semidefinite cone.
 5. The exponential cone

   .. math::

        (u,v,w) \in K_e \doteq \mathrm{cl}\{(x,y,z) |  z \geq y \exp(x/y), y>0\}.

The precise nature of the vectorized positive semidefinite cone is a little delicate, is
covered later.
For now it's useful to say that the ``Awesome(ConicSolver)`` class will access an
explicit representation for problem :math:`(P)` in in ``apply``, with a code snippet like

.. code::

    # from cvxpy.constraints import Zero, NonNeg, SOC, PSD, ExpCone
    #  ...
    if not problem.formatted:
        problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
    constr_map = problem.constr_map
    cone_dims = problem.cone_dims
    c, d, A, b = problem.apply_parameters()

The variable ``constr_map`` is is a dict of lists of CVXPY Constraint objects.
The dict is keyed by the references to CVXPY's Zero, NonNeg, SOC, PSD, and
ExpCone classes. You will need to interact with these constraint classes during
dual variable recovery.
For the other variables in that code snippet ...
 -  ``c, d`` define the objective function ``c @ x + d``, and
 - ``A, b, cone_dims`` define the abstractions :math:`A`, :math:`b`,
   :math:`K` in problem  :math:`(P)`.

The first step in writing a solver interface is to understand the exact
meanings of ``A, b, cone_dims``, so that you can correctly build a primal
problem using the third-party ``AwesomePy`` interface to the *Awesome* solver.
The ``cone_dims`` object is an instance of the ConeDims class, as defined in
`cone_matrix_stuffing.py
<https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py>`_;
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
   Here ``cone_dims.psd[0]`` gives the *order* of the first PSD cone.
   So if ``cone_dims.psd[0] == 5``, then this constraint involves the next
   ``num_rows = 5*(5+1)//2`` rows of ``A, b``.
   CVXPY uses the same vectorization for the PSD cone as the SCS solver.
   It might help to reference the functions ``tri_to_full`` and ``scs_psd_vec_to_psd_mat`` in
   `scs_conif.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/scs_conif.py>`_
   to see how the vectorized form of a PSD matrix compares to its full, square form.
 - The last block of ``3 * cone_dims.exp`` rows in ``A, b`` correspond to consecutive
   three-dimensional exponential cones, as defined by :math:`K_e` above.

If *Awesome* supports nonlinear constraints like SOC, ExpCone, or PSD, then it's possible
that you will need to transform data ``A, b`` in order to write these constraints in
the form expected by ``AwesomePy``.
The most common situations are when ``AwesomePy`` parametrizes the second-order cone
as :math:`K = \{ (x,t) \,:\, \|x\|\leq t \} \subset \mathbb{R}^n \times \mathbb{R}`,
or when it parametrizes :math:`K_e \subset \mathbb{R}^3` as some permutation of
what we defined earlier.


Dual variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual variable extraction should be handled in ``Awesome(ConicSolver).invert``.
To perform this step correctly, it's necessary to consider how CVXPY forms
a Lagrangian for the primal problem :math:`(P)`.
Let's say that the affine map :math:`Ax + b` in the feasible set
:math:`Ax + b \in K \subset \mathbb{R}^m` is broken up into five blocks of sizes
:math:`m_1,\ldots,m_5` where the blocks correspond (in order) to zero-cone, nonnegative cone,
second-order cone, vectorized PSD cone, and exponential cone constraints.
Then CVXPY defines the dual to :math:`(P)` by forming a Lagrangian

.. math::

    \mathcal{L}(x,\mu_1,\ldots,\mu_5) = c^T x - \sum_{i=i}^5 \mu_i^T (A_i x + b_i)

in dual variables :math:`\mu_1 \in \mathbb{R}^{m_1}`, :math:`\mu_2 \in \mathbb{R}^{m_2}_+`,
and :math:`\mu_i \in K_i^* \subset \mathbb{R}^{m_i}` for :math:`i \in \{3,4,5\}`.
Here, :math:`K_i^*` denotes the dual cone to :math:`K_i` under the standard inner product.

More remarks on dual variables (particularly SOC dual variables) can be found in
`this comment on a GitHub thread <https://github.com/cvxgrp/cvxpy/issues/948#issuecomment-592781675>`_.

Most concrete implementations of the ConicSolver class use a common set of helper
functions for dual variable recovery, found in
`reductions/solvers/utilities.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/utilities.py>`_.
Refer to `MOSEK(ConicSolver).invert
<https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/mosek_conif.py#L479>`_
for a well documented and less abstract implementation of dual variable recovery.

Registering a solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Correctly implementing ``Awesome(ConicSolver)`` isn't enough to call *Awesome* from CVXPY.
You need to make edits in a handful of other places, namely

 - `conic_solvers/__init__.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/conic_solvers/__init__.py>`_,
 - `solvers/defines.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/reductions/solvers/defines.py>`_, and
 - `cvxpy/__init__.py <https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/__init__.py>`_.

The existing content of those files should make it clear what's needed
to add *Awesome* to CVXPY.

Writing tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for  ``Awesome(ConicSolver)`` should be placed in `cvxpy/tests/test_conic_solvers.py
<https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/tests/test_conic_solvers.py>`_.
The overwhelming majority of tests in that file only take a single line, because
we make consistent use of a general testing framework defined in
`solver_test_helpers.py
<https://github.com/cvxgrp/cvxpy/blob/master/cvxpy/tests/solver_test_helpers.py>`_.
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
call modifed versions of a test with different solver parameters, for example

.. code::

    def test_mosek_lp_1(self):
        # default settings
        StandardTestLPs.test_lp_1(solver='MOSEK')  # 4 places
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='MOSEK', places=6, bfs=True)



.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _CVXOPT: http://cvxopt.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/