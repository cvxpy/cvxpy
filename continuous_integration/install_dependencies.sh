#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e
uv venv
if [[ "$RUNNER_OS" == "Windows" ]]; then
  . .venv/Scripts/activate
else
  . .venv/bin/activate
fi

#if  [[ "$RUNNER_OS" == "Linux" ]]; then 
#  sudo apt install openblas
#fi

uv pip install pytest pytest-cov hypothesis "setuptools>65.5.1"

uv pip install scs clarabel osqp

if [[ "$RUNNER_OS" != "macOS" ]]; then
  uv pip install mkl
fi

uv pip install scipy numpy

#if [[ "$USE_OPENMP" == "True" ]]; then
  #uv pip install openmp
#fi
