#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment
# variables defined in the build.yml in .github/workflows/.

set -e

# Installs packages that need BLAS/LAPACK. On Windows, packages with no wheel for this
# platform (e.g. ARM64, free-threaded) are built against the official OpenBLAS binaries
# and repaired with delvewheel so the OpenBLAS DLL is vendored.
install_blas_packages() {
  if [[ "$RUNNER_OS" != "Windows" ]]; then
    uv pip install "$@"
    return
  fi
  local missing=()
  for spec in "$@"; do
    uv pip install --only-binary :all: "$spec" || missing+=("$spec")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    return
  fi

  uv pip install pip delvewheel
  local version=0.3.34 asset sha root
  if [[ "$RUNNER_ARCH" == "ARM64" ]]; then
    asset=woa64-dll sha=b309f6ce5961629231d1575cee6e1a4793342d21ed233447462ed534bf3fcd80
    root=build/openblas/OpenBLAS
  else
    asset=x64 sha=e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b
    root=build/openblas
  fi
  rm -rf build/openblas
  mkdir -p build/openblas
  curl -fsSL -o build/openblas/openblas.zip \
    "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$version/OpenBLAS-$version-$asset.zip"
  echo "$sha  build/openblas/openblas.zip" | sha256sum -c -
  python -m zipfile -e build/openblas/openblas.zip build/openblas

  if [[ "$RUNNER_ARCH" != "ARM64" ]]; then
    # The x64 (MinGW) archive lacks the OpenBLAS::OpenBLAS target and an openblas.lib.
    cat >> "$root/lib/cmake/openblas/OpenBLASConfig.cmake" <<'EOF'
add_library(OpenBLAS::OpenBLAS SHARED IMPORTED)
set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
  IMPORTED_IMPLIB "${_OpenBLAS_ROOT_DIR}/lib/libopenblas.lib"
  IMPORTED_LOCATION "${_OpenBLAS_ROOT_DIR}/bin/libopenblas.dll"
  INTERFACE_INCLUDE_DIRECTORIES "${_OpenBLAS_ROOT_DIR}/include")
EOF
    cp "$root/lib/libopenblas.lib" "$root/lib/openblas.lib"
  fi

  # Backslash paths stop Git Bash from rewriting them when passed through env vars.
  root=$(cygpath -w "$PWD/$root")
  for spec in "${missing[@]}"; do
    local args=()
    # The runners put MinGW gcc on PATH, which meson picks over MSVC unless --vsenv is set.
    if [[ "$spec" == scs* ]]; then
      args=(-Csetup-args=--vsenv)
    fi
    LIB="$root\\lib;${LIB:-}" CMAKE_PREFIX_PATH="$root" \
      python -m pip wheel --no-deps -w build/openblas/dist "${args[@]}" "$spec"
  done
  delvewheel repair --add-path "$root\\bin" -w build/openblas/wheelhouse build/openblas/dist/*.whl
  uv pip install build/openblas/wheelhouse/*.whl
}

uv venv
if [[ "$RUNNER_OS" == "Windows" ]]; then
  . .venv/Scripts/activate
else
  . .venv/bin/activate
fi

uv pip install pytest pytest-cov hypothesis "setuptools>65.5.1"

uv pip install clarabel osqp highspy

# Keep the sparsediffpy spec in sync with pyproject.toml.
install_blas_packages scs "sparsediffpy>=0.7.0,<0.8.0"

# mkl only publishes x86_64 Linux and Windows wheels.
if [[ "$RUNNER_OS" != "macOS" && "$RUNNER_ARCH" != "ARM64" ]]; then
  uv pip install mkl
fi

uv pip install scipy numpy

#if [[ "$USE_OPENMP" == "True" ]]; then
  #uv pip install openmp
#fi
