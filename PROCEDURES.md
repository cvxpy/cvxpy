# Procedures for making a new CVXPY release

This file provides the procedures for releasing a new version of CVXPY.
The process involves defining a new release in the commit history,
packaging and deploying the updated source code, and deploying updated 
web documentation.

## Defining a new release

CVXPY's `setup/versioning.py` file defines the following *versioning data*
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
2. The versioning data in `setup/versioning.py` should already be
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
5. Update `docs/source/index.rst` to say "Welcome to CVXPY 1.3"
6. Extend the ``version_info`` field in ``doc/source/conf.py``.

### Incrementing the MICRO version number (a.k.a., releasing a patch)

Let's say we're releasing CVXPY 1.2.1

1. Create a new branch `patch/1.2.1` from `release/1.2.x`. Go through all commits merged into the master branch since the previous release and use `git cherry-pick abc123`, where `abc123` is the commit into the master branch. Create a pull request against the `release/1.2.x` branch listing the commits contained in the patch.
2. Starting from ``release/1.2.x``, the versioning data in `setup/versioning.py` (`setup.py` in earlier releases) should already be
   ```
   MAJOR = 1
   MINOR = 2
   MICRO = 1
   IS_RELEASED = False
   IS_RELEASE_BRANCH = True
   ```
   Change ``IS_RELEASED = True`` and commit that change with
   the tag ``v1.2.1``.
3. Lay the groundwork for the next release on this branch.
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

## Creating a release on GitHub
Go to the [Releases](https://github.com/cvxpy/cvxpy/releases) tab and click "Draft a new release". Select the previously created tag and write release notes. For minor releases, this includes a summary of new features and deprecations. Additionally, we mention the PRs contained in the release and their contributors. Take care to select the "set as the latest release" only for minor releases or patches to the most recent major release.

## Deploying updated documentation to gh-pages

The web documentation is built and deployed using a GitHub action that can be found [here](https://github.com/cvxpy/cvxpy/blob/master/.github/workflows/docs.yml).

To deploy the docs for a specific version, navigate to the [actions tab](https://github.com/cvxpy/cvxpy/actions) and select the `docs` workflow.
Under `Use workflow from`, select the **Tags** tab and choose the version you want to deploy the docs for.
This builds the docs and commits them to the `gh-pages` branch. This in turn triggers the deployment through the `github-pages bot`, which can also be monitored in the [actions tab](https://github.com/cvxpy/cvxpy/actions).

After the deployment, make sure that the docs are accessible through the browser, and the version selector displays all expected versions.

