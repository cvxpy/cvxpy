# Contributing to CVXPY

This document is a guide to contributing to CVXPY.

We welcome all contributions. You don't need to be an expert in optimization
to help out.

## Checklist

Contributions are made through
[pull requests](https://help.github.com/articles/using-pull-requests/).
Before sending a pull request, make sure you do the following:
- Check that your code adheres to our [coding style](#code-style)
- [Write unit tests](#writing-unit-tests)
- Run the [unit tests](#running-unit-tests) and check that they're passing
- Run the [benchmarks](#benchmarks) to make sure your change does not introduce a regression

## Building CVXPY from source

You'll need to build CVXPY locally in order to start editing code. We recommend
that you do this in a fresh [virtual
environment](https://virtualenv.pypa.io/en/latest/).

To install CVXPY from source, clone the Github repository, navigate to the
repository root, and run the following command:

```bash
make develop
```

Here we install uv, create a virtual environment and install the dependencies
declared in 'setup.py' within.

## Contributing code

To contribute to CVXPY, send us pull requests. For those new to contributing,
check out Github's
[guide](https://help.github.com/articles/using-pull-requests/).

Once you've made your pull request, a member of the CVXPY development team
will assign themselves to review it. You might have a few back-and-forths
with your reviewer before it is accepted, which is completely normal. Your
pull request will trigger continuous integration tests for many different
Python versions and different platforms. If these tests start failing, please
fix your code and send another commit, which will re-trigger the tests.

If you'd like to add a new feature to CVXPY, or a new example to our
[library](https://www.cvxpy.org/examples/index.html), please do propose your
change on a Github issue, to make sure that your priorities align with ours.

If you'd like to contribute code but don't know where to start, try one of the
following:

* Read the CVXPY source and enhance the documentation, or address TODOs
* Browse the [issue tracker](https://github.com/cvxpy/cvxpy/issues), and
  look for the issues tagged "help wanted".
* Polish the [example library](https://www.cvxpy.org/examples/index.html) or add new examples
* Add a [benchmark](https://github.com/cvxpy/cvxpy/tree/master/cvxpy/tests/test_benchmarks.py)

## Code style

The package [pre-commit](https://pre-commit.com/) can be installed 
to check all our conventions automatically before every commit.

```
make pre-commit
```

Note on isort: if you see files changed by isort that you did not actively make
changes to as part of your PR, it is an indication that the config was not 
picked up correctly. See [issue 1653](https://github.com/cvxpy/cvxpy/issues/1653) for details.

## Writing unit tests

Most code changes will require new unit tests. (Even bug fixes require unit tests,
since the presence of bugs usually indicates insufficient tests.) CVXPY tests
live in the directory `cvxpy/tests`, which contains many files, each of which
contains many unit tests. When adding tests, try to find a file in which your
tests should belong; if you're testing a new feature, you might want to create
a new test file.

We use the standard Python [`unittest`](https://docs.python.org/3/library/unittest.html)
framework for our tests. Tests are organized into classes, which inherit from
`BaseTest` (see `cvxpy/tests/base_test.py`). Every method beginning with `test_` is a unit
test.

## Running unit tests

We use `pytest` to run our unit tests, which you can install with `pip install pytest hypothesis`.
To run all unit tests, `cd` into `cvxpy/tests` and run the following command:

```bash
pytest
````

To run tests in a specific file (e.g., `test_dgp.py`), use

```bash
pytest test_dgp.py
```

To run a specific test method (e.g., `TestDgp.test_product`), use

```bash
pytest test_dgp.py::TestDgp::test_product
```

Please make sure that your change doesn't cause any of the unit tests to fail.

`pytest` suppresses stdout by default. To see stdout, pass the `-s` flag
to `pytest`.

## Benchmarks

CVXPY has a few benchmarks in `cvxpy/tests/test_benchmarks.py`, which test
the time to canonicalize problems. Please run

```
pytest -s cvxpy/tests/test_benchmarks.py
```

with and without your change, to make sure no performance regressions are
introduced. If you are making a code contribution, please include the output of
the above command (with and without your change) in your pull request.
