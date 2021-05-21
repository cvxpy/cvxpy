#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e

if [[ "$RUNNER_OS" == "Linux" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -qq gfortran libgfortran3
fi

if [[ "$PYTHON_VERSION" == "3.6" ]]; then
  conda install mkl pip pytest scipy=1.1 numpy=1.15 lapack ecos scs osqp flake8 cvxopt
elif [[ "$PYTHON_VERSION" == "3.7" ]]; then
  conda install mkl pip pytest scipy=1.1 numpy=1.15 lapack ecos scs osqp flake8 cvxopt
elif [[ "$PYTHON_VERSION" == "3.8" ]]; then
  # There is a config that works with numpy 1.14, but not 1.15!
  # So we fix things at 1.16.
  # Assuming we use numpy 1.16, the earliest version of scipy we can use is 1.3.
  conda install mkl pip pytest scipy=1.3 numpy=1.16 lapack ecos scs osqp flake8 cvxopt
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.19.
  # Given numpy 1.19, the earliest version of scipy we can use is 1.5.
  conda install mkl pip pytest scipy=1.5 numpy=1.19 lapack ecos scs flake8 cvxopt
  python -m pip install osqp
fi

if [[ "$USE_OPENMP" == "True" ]]; then
    conda install -c conda-forge openmp
fi

python -m pip install diffcp

if [[ "$COVERAGE" == "True" ]]; then
    python -m pip install coverage coveralls
fi
