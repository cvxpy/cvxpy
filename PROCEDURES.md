# Procedures for making a new CVXPY release

This file provides the procedures for releasing a new version of CVXPY.
The process involves defining a new release in the commit history,
packaging and deploying the updated source code, and deploying updated 
web documentation.

## Defining a new release

The version is derived from git tags by [setuptools-scm](https://setuptools-scm.readthedocs.io/)
(configured in `pyproject.toml`). There is no hand-edited version constant.
At build time, setuptools-scm writes `cvxpy/_version.py`, which is re-exported
as `cvxpy.__version__`.

- Commits *exactly* on a `vX.Y.Z` tag produce the clean version `X.Y.Z`.
- Any other commit produces a dev version like `X.Y.Z.devN+g<sha>`, where the
  base `X.Y.Z` is derived from the most recent ancestor tag using the
  `semver-pep440-release-branch` scheme (minor bump on `master`, patch bump on
  `release/*` branches).

For this scheme to work, **every minor release tag must be reachable from
`master`**. Patch tags live on `release/*` branches (where they only need to
be reachable from that branch). Minor-release tags must be placed on the
`master` commit that the release branch was cut from.

### Incrementing the MINOR version number

Let's say we're releasing 1.10.0.

1. On ``master``, at the commit you want to release from, create an annotated
   tag:
   ```
   git tag -a v1.10.0 -m "Release 1.10.0"
   ```
2. Cut the release branch from that same commit and push both the branch and
   the tag:
   ```
   git checkout -b release/1.10.x v1.10.0
   git push origin release/1.10.x
   git push origin v1.10.0
   ```
   Pushing the tag triggers the PyPI deploy via `.github/workflows/build.yml`.
3. Update `doc/source/index.rst` to say "Welcome to CVXPY 1.11" and extend the
   ``version_info`` field in ``doc/source/conf.py``. Commit these to ``master``.

Master now reports `1.11.0.devN+g<sha>` as its version (the next minor),
because the most recent ancestor tag is `v1.10.0` and the scheme bumps the
minor on non-release branches.

### Incrementing the MICRO version number (a.k.a., releasing a patch)

Let's say we're releasing CVXPY 1.10.1.

1. Create a new branch `patch/1.10.1` from `release/1.10.x`. Cherry-pick the
   relevant commits from ``master`` with `git cherry-pick abc123`, and open a
   pull request against `release/1.10.x` listing the commits in the patch.
2. After the PR merges, tag the resulting commit on `release/1.10.x`:
   ```
   git tag -a v1.10.1 -m "Release 1.10.1"
   git push origin v1.10.1
   ```
   Pushing the tag triggers the PyPI deploy.

Patch tags do not need to be merged or replicated onto `master` — master only
cares about the most recent *minor* tag (`v1.10.0`) for its own dev versions.

## Deploying a release to PyPI

Deployments to PyPI are automatically triggered for every tagged commit of the release process described above.
This workflow is defined as a GitHub action and can be found [here](https://github.com/cvxpy/cvxpy/blob/master/.github/workflows/build.yml).
The progress of the deploy can be inspected by opening the workflow run marked with `v*` from the [actions tab](https://github.com/cvxpy/cvxpy/actions).

After a successful deployment, the result should be verified on PyPI.
In particular, for both [cvxpy](https://pypi.org/project/cvxpy/) and [cvxpy-base](https://pypi.org/project/cvxpy-base/) 
source files as well as all expected wheel files should be present.
The `cvxpy-base` files should include the Pyodide wheel artifact built by the `build_pyodide_wheels` job.

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

To generate the list of PRs and contributors, use the `tools/release_notes.py` script:
```
python tools/release_notes.py v1.8.0  # minor release
python tools/release_notes.py v1.7.5  # patch release
```
For minor releases, the script automatically excludes PRs that were cherry-picked into the previous release branch's patch releases. For patch releases, it compares against the previous patch tag.

## Deploying updated documentation to gh-pages

The web documentation is built and deployed using a GitHub action that can be found [here](https://github.com/cvxpy/cvxpy/blob/master/.github/workflows/docs.yml).

To deploy the docs for a specific version, navigate to the [actions tab](https://github.com/cvxpy/cvxpy/actions) and select the `doc_deploy` workflow.
Under `Use workflow from`, select the **Tags** tab and choose the version you want to deploy the docs for.
This builds the docs and commits them to the `gh-pages` branch. This in turn triggers the deployment through the `github-pages bot`, which can also be monitored in the [actions tab](https://github.com/cvxpy/cvxpy/actions).

After the deployment, make sure that the docs are accessible through the browser, and the version selector displays all expected versions.
