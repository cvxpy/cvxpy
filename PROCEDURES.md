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
2. The versioning data in setup.py should already be
   ```
   MAJOR = 1
   MINOR = 2
   MICRO = 0
   IS_RELEASED = False
   IS_RELEASE_BRANCH = False
   ```
   Set ``IS_RELEASE_BRANCH = True`` and ``IS_RELEASED = True``.
   Commit these changes and tag the commit as ``v1.2.0``.
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

1. Starting from ``release/1.2.x``, the versioning data in setup.py should already be
   ```
   MAJOR = 1
   MINOR = 2
   MICRO = 1
   IS_RELEASED = False
   IS_RELEASE_BRANCH = True
   ```
   Change ``IS_RELEASED = True`` and commit that change with
   the tag ``v1.2.1``.
2. Lay the groundwork for the next release on this branch.
   Do this by setting ``MICRO = 2``, ``IS_RELEASED = False``, and 
   committing those changes.
   *Do not* tag the commit as a release.
   The state of this branch is effectively a pre-release of 
   CVXPY 1.2.2.

## Deploying a release to PyPI

Deployments to PyPI are automatically triggered for every tagged commit of the release process described above.
This workflow is defined as a GitHub action and can be found [here](https://github.com/cvxpy/cvxpy/blob/master/.github/workflows/build.yml).
The progress of the deploy can be inspected by opening the workflow run marked with `v*` from the [actions tab](https://github.com/cvxpy/cvxpy/actions).

After a successful deployment, the result should be verified on PyPI.
In particular, for both [cvxpy](https://pypi.org/project/cvxpy/) and [cvxpy-base](https://pypi.org/project/cvxpy-base/) 
source files as well as all expected wheel files should be present.

If the action fails intermittently, e.g., because of time-outs during the installation of the dependencies, it can be retriggered from the [actions tab](https://github.com/cvxpy/cvxpy/actions).
If changes are required, the `DEPLOY` variable needs to be set manually in the workflow to allow deploys from a non-tagged commit.


## Deploying a release to conda-forge

The following remarks are based on [@h-vetinari's comment on this GitHub Pull Request](https://github.com/cvxpy/cvxpy/pull/1598#discussion_r787062572).

Upon creating a tagged commit in the cvxpy repo, a bot will open an upgrade PR on [cvxpy's conda-forge feedstock](https://github.com/conda-forge/cvxpy-feedstock).
All necessary changes will be concentrated in recipe/meta.yaml.
The changes include 
 1. updating dependency requirements,
 2. updating the version number, and 
 3. adding the hash of the sources.

The conda-forge bot will handle (2) and (3) automatically.
Any changes to (1) require manual intervention but are rare.

Once the PR is opened the conda-forge bot will build the packages and run the full test suite
(except for cross-compiled architectures like osx-arm).
If there are failures, then the PR is not mergeable
(resp. no artefacts will be uploaded for failing jobs once merged).
The updated cvxpy release will be available on conda-forge after the PR is merged.
Merging PRs needs maintainership rights on the feedstock (which several cvxpy-people have)

If issues come up then we can ask h-vetinari or conda-forge/core for help,
although we should only ping conda-forge/core if h-vetinari is unavailable.

An import note: cvxpy's conda-forge feedstock includes a patch to remove ``pyproject.toml``,
because it ignores and tramples over the required build dependencies as conda-forge sets them up.
If this file has changed between versions, the old patch will fail to apply and will need to be rebased.


## Deploying updated documentation to gh-pages
