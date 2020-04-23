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

Solver interfaces
----------------------

Third-party numerical optimization solvers are the lifeblood of CVXPY.
We are very grateful to anyone who would be willing to volunteer their time to
improve our existing solver interfaces, or create interfaces to new solvers.
Improving an existing interface can usually be handled like fixing a bug.
Creating a new interface requires much more work, and warrants coordination
with CVXPY principal developers before writing any code.

This section of the contributing guide outlines considerations when adding new solver interfaces.

QPSolver vs ConicSolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All solver interfaces are contained in ``cvxpy/reductions/solvers/``,
which specializes into code paths for QPs and cone programs.
You need to create a python file ``awesome_solver.py`` which defines
an appropriate class: ``AwesomeSolver(QPSolver)`` or ``AwesomeSolver(ConicSolver)``.

From CVXPY's perspective, the only difference between a QP solver and a conic solver
is that a QP solver allows for a quadratic objective, while a conic solver requires
a linear objective.
Conic solvers generally allow for more complicated constraints compared to QP solvers.
In the case of some commercial packages (such as CPLEX), CVXPY maintains separate interface files
for QPs versus conic programs.

Essential functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain ``.apply()``, ``.solve_via_data()``, and ``.invert()``.

Address differences in how someone would implement ``.apply()`` depending on if they
had a QP solver versus a conic solver.
For example: QP solvers inherit their apply function from ``QPSolver.apply``,
but conic solvers must implement ``.apply()`` from scratch.


Dual variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain that this logic is usually stored in ``.invert()``, and that correctly
implementing this logic requires a good understanding of what happened in ``.apply()``.

Explain CVXPY's conventions for dual variables, and assumptions on the Lagrangian.
Incorporate some of `my comment <https://github.com/cvxgrp/cvxpy/issues/923#issuecomment-590516011>`_ or
`my other comment <https://github.com/cvxgrp/cvxpy/issues/948#issuecomment-592781675>`_.

Passing solver options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain how this fits in to ``.solve_via_data()``.

Writing tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explain the conic solver testing framework.
Explain the QP solver testing framework.

Maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* We doesn't ship major releases with broken solver interfaces.
  If a change to cvxpy breaks a solver interface and no one steps up to fix it,
  there's a chance that the interface could be removed.

* The core development team is generally willing to spend some effort on fixing a solver interface before removing it.
  However if we encounter a notable obstacle, the solver might be dropped. Examples of notable obstacles include:

  * The solver doesn't have an interface for a sufficiently high version of python.
    SuperSCS was dropped because it only had a python 2.7 interface, and cvxpy 1.1 requires python >= 3.5

  * The solver has been deprecated by its principal developer. Elemental fell into this category.

  * The underling numerical implementation of the solver has standing correctness issues, and routinely
    causes confusion for CVXPY users. ECOS_BB fell into this category.


.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _CVXOPT: http://cvxopt.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.scipy.org/