#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e

conda config --set remote_connect_timeout_secs 30.0
conda config --set remote_max_retries 10
conda config --set remote_backoff_factor 2
conda config --set remote_read_timeout_secs 120.0

if [[ "$PYTHON_VERSION" == "3.7" ]] || [[ "$PYTHON_VERSION" == "3.8" ]]; then
  conda install scipy=1.3 numpy=1.16 mkl pip pytest pytest-cov lapack ecos scs osqp cvxopt setuptools=63.4.2
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.19.
  # Given numpy 1.19, the earliest version of scipy we can use is 1.5.
  conda install scipy=1.5 numpy=1.19 mkl pip pytest lapack ecos scs osqp cvxopt setuptools=64.0.2
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    # The earliest version of numpy that works is 1.21.
    # Given numpy 1.21, the earliest version of scipy we can use is 1.7.
    conda install scipy=1.7 numpy=1.21 mkl pip pytest lapack ecos scs osqp cvxopt setuptools=64.0.2
fi

python -m pip install "ortools>=9.3,<9.5" coptpy cplex sdpa-python diffcp gurobipy xpress clarabel

# cylp has wheels for all versions 3.7 - 3.10, except for 3.7 on Windows
if [[ "$PYTHON_VERSION" != "3.7" ]] && [[ "$RUNNER_OS" != "Windows" ]]; then
  python -m pip install cylp
fi

# SCIP only works with scipy >= 1.5 due to dependency conflicts when installing on Linux/macOS
if [[ "$PYTHON_VERSION" == "3.9" ]] || [[ "$RUNNER_OS" == "Windows" ]]; then
  conda install pyscipopt"<4.0"  # TODO: update interface https://github.com/cvxpy/cvxpy/pull/1628
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]]; then
    python -m pip install mosek
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  conda install -c conda-forge openmp
fi
