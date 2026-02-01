# diff_engine_core Release Procedures

This document describes how to create releases for diff_engine_core and update the DNLP submodule. All commands assume you start from the DNLP repository root.

## Current State

- **Current version**: v0.1.0
- **Submodule commit**: ee7fed2
- **Note**: Python bindings are maintained in DNLP at `cvxpy/.../diff_engine/_bindings/`, not in diff_engine_core. The C library is now a pure C project.

## Creating a New Release

1. **Update version in `diff_engine_core/CMakeLists.txt`**:
   ```cmake
   set(DIFF_ENGINE_VERSION_MAJOR 0)
   set(DIFF_ENGINE_VERSION_MINOR 2)
   set(DIFF_ENGINE_VERSION_PATCH 0)
   ```

2. **Commit the version bump and create tag**:
   ```bash
   cd diff_engine_core
   git add CMakeLists.txt
   git commit -m "Bump version to 0.2.0"
   git push origin main
   git tag v0.2.0
   git push origin v0.2.0
   cd ..
   ```

3. **Automatic release**: GitHub Actions will:
   - Run tests on the tagged commit (ubuntu-latest, macos-latest)
   - Create a GitHub release with auto-generated release notes

## Updating DNLP Submodule

After creating a release in diff_engine_core, update this repository:

1. **Update submodule to the new tag**:
   ```bash
   cd diff_engine_core
   git fetch --tags
   git checkout v0.2.0
   cd ..
   ```

2. **Commit the submodule update**:
   ```bash
   git add diff_engine_core
   git commit -m "Update diff_engine_core to v0.2.0"
   git push
   ```

## Versioning Scheme

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes (function signatures, struct layouts)
- **MINOR**: New features, backward compatible (new atoms, optimizations)
- **PATCH**: Bug fixes only

## Verifying a Release

After pushing a tag, verify the release was created:

1. Check [GitHub Actions](https://github.com/dance858/DNLP-Differentiation-Engine/actions) for the release workflow status
2. Check [Releases](https://github.com/dance858/DNLP-Differentiation-Engine/releases) for the new release entry
3. Test the submodule update:
   ```bash
   cd diff_engine_core
   git checkout v0.2.0
   cd ..
   git status  # Should show submodule change
   ```
