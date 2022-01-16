# Procedures for making a new CVXPY release

This file provides the procedures for releasing a new version of CVXPY.
The process involves defining a new release in the commit history,
packaging and deploying the updated source code, and deploying updated 
web documentation.

## Defining a new release

CVXPY's setup.py file defines the following *versioning data*
   ```
   MAJOR : an int
   MINOR : an int
   MICRO : an int
   IS_RELEASED : a bool
   IS_RELEASE_BRANCH : a bool
   ```
Here we give the procedure for maintaining these values
as one makes new minor and micro releases.

### Incrementing the MINOR version number

Let's say we're releasing 1.2.0.

1. Starting from ``master``, checkout a new branch called ``release/1.2.x``.
2. The versioning data in setup.py should be
   ```
   MAJOR = 1
   MINOR = 2
   MICRO = 0
   IS_RELEASED = False
   IS_RELEASE_BRANCH = False
   ```
   Set ``IS_RELEASE_BRANCH = True`` and ``IS_RELEASED = True``.
   Commit these changes and tag the commit as the release of CVXPY 1.2.0.
3. Lay the groundwork for the next release on this branch.
   Do this by setting ``MICRO = 1``, ``IS_RELEASED = False``, and
   committing those changes.
   *Do not* tag the commit as a release.
   The state of this branch is effectively a pre-release of 
   CVXPY 1.2.1.
4. Checkout ``master``. Change the versioning data 
   from ``MINOR = 2`` to ``MINOR = 3`` and commit.
   The state of this branch is effectively a pre-release of
   CVXPY 1.3.0.

### Incrementing the MICRO version number (a.k.a., releasing a patch)

Let's say we're releasing CVXPY 1.2.1

1. Starting from ``release/1.2.x``, the versioning data in setup.py should be
   ```
   MAJOR = 1
   MINOR = 2
   MICRO = 1
   IS_RELEASED = False
   IS_RELEASE_BRANCH = True
   ```
   Change ``IS_RELEASED = True`` and commit that change.
   Tag the commit as the release of CVXPY 1.2.1.
2. Lay the groundwork for the next release on this branch.
   Do this by setting ``MICRO = 2``, ``IS_RELEASED = False``, and 
   committing those changes.
   *Do not* tag the commit as a release.
   The state of this branch is effectively a pre-release of 
   CVXPY 1.2.2.

## Deploying a release to PyPI

## Deploying a release to conda-forge

## Deploying updated documentation to gh-pages