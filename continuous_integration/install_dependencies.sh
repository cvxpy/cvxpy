#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e

if [[ "$PYTHON_VERSION" == "3.6" ]] || [[ "$PYTHON_VERSION" == "3.7" ]] || [[ "$PYTHON_VERSION" == "3.8" ]]; then
  conda install scipy=1.3 numpy=1.16 mkl pip pytest lapack ecos scs osqp cvxopt
  python -m pip install cplex  # CPLEX is not available yet on 3.9
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.19.
  # Given numpy 1.19, the earliest version of scipy we can use is 1.5.
  conda install scipy=1.5 numpy=1.19 mkl pip pytest lapack ecos scs osqp cvxopt
fi

# CBC comes with wheels for windows and needs coin-or-cbc to compile otherwise
# conda-forge in progress: https://github.com/conda-forge/staged-recipes/pull/14950
if [[ "$RUNNER_OS" != "Windows" ]]; then
  conda install coin-or-cbc
fi
if [[ "$PYTHON_VERSION" != "3.6" ]] && [[ "$RUNNER_OS" != "Windows" ]]; then
  python -m pip install cylp
fi

# SCIP only works with scipy >= 1.5 due to dependency conflicts when installing on Linux/macOS
if [[ "$PYTHON_VERSION" == "3.9" ]] || [[ "$RUNNER_OS" == "Windows" ]]; then
  conda install pyscipopt
fi

python -m pip install diffcp gurobipy xpress

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]]; then
    python -m pip install mosek
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  conda install -c conda-forge openmp
fi

if [[ "$COVERAGE" == "True" ]]; then
  python -m pip install coverage coveralls
fi
