#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e

conda config --set remote_connect_timeout_secs 30.0
conda config --set remote_max_retries 10
conda config --set remote_backoff_factor 2
conda config --set remote_read_timeout_secs 120.0

# Issue with installing setuptools > 65.5.1 through conda on mac with Python 3.7.
if [[ "$PYTHON_VERSION" == "3.7" ]] && [[ "$RUNNER_OS" == "macos-11" ]]; then
  conda install scipy=1.3 numpy=1.16 mkl pip pytest pytest-cov lapack ecos scs osqp cvxopt proxsuite setuptools
elif [[ "$PYTHON_VERSION" == "3.7" ]] || [[ "$PYTHON_VERSION" == "3.8" ]]; then
  conda install scipy=1.3 numpy=1.16 mkl pip pytest pytest-cov lapack ecos scs osqp cvxopt proxsuite "setuptools>65.5.1"
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.19.
  # Given numpy 1.19, the earliest version of scipy we can use is 1.5.
  conda install scipy=1.5 numpy=1.19 mkl pip pytest lapack ecos scs osqp cvxopt proxsuite "setuptools>65.5.1"
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    # The earliest version of numpy that works is 1.21.
    # Given numpy 1.21, the earliest version of scipy we can use is 1.7.
    conda install scipy=1.7 numpy=1.21 mkl pip pytest lapack ecos scs osqp cvxopt proxsuite "setuptools>65.5.1"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    # The earliest version of numpy that works is 1.23.4.
    # Given numpy 1.23.4, the earliest version of scipy we can use is 1.9.3.
    conda install scipy=1.9.3 numpy=1.23.4 mkl pip pytest lapack ecos scs osqp cvxopt proxsuite "setuptools>65.5.1"
fi


# Python 3.8 on Windows and Linux will uninstall NumPy 1.16 and install NumPy 1.24 without the exception.
if [[ "$PYTHON_VERSION" == "3.8" ]] && [[ "$RUNNER_OS" != "macos-11" ]]; then
  python -m pip install gurobipy clarabel
else
  python -m pip install "ortools>=9.3,<9.5" coptpy cplex diffcp gurobipy xpress clarabel
  if [[ "$RUNNER_OS" == "Windows" ]]; then
    python -m pip install sdpa-multiprecision
  else
    python -m pip install sdpa-python
  fi
fi

# cylp has wheels for all versions 3.7 - 3.10, except for 3.7 on Windows
if [[ "$PYTHON_VERSION" != "3.7" ]] && [[ "$PYTHON_VERSION" != "3.11" ]] && [[ "$RUNNER_OS" != "Windows" ]]; then
  python -m pip install cylp
fi

# SCIP only works with scipy >= 1.5 due to dependency conflicts when installing on Linux/macOS
if [[ "$PYTHON_VERSION" == "3.9" ]] || [[ "$RUNNER_OS" == "Windows" ]]; then
  conda install pyscipopt
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]] && [[ "$PYTHON_VERSION" != "3.11" ]]; then
    python -m pip install mosek
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  conda install -c conda-forge openmp
fi
