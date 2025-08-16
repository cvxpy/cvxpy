#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e
uv venv
. .venv/bin/activate

#if  [[ "$RUNNER_OS" == "Linux" ]]; then 
#  sudo apt install openblas
#fi

uv pip install pytest pytest-cov hypothesis "setuptools>65.5.1"

uv pip install scs clarabel osqp

if [[ "$RUNNER_OS" != "macOS" ]]; then
  uv pip install mkl
fi

# Install newest stable versions for Python 3.13.
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
  uv pip install scipy numpy
else
  uv pip install scipy=1.13.0 numpy=1.26.4
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  uv pip install openmp
fi
