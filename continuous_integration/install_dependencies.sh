#!/bin/bash
# This script is meant to be called by the "install" step defined in
# build.yml. The behavior of the script is controlled by environment 
# variables defined in the build.yml in .github/workflows/.

set -e

conda config --set remote_connect_timeout_secs 30.0
conda config --set remote_max_retries 10
conda config --set remote_backoff_factor 2
conda config --set remote_read_timeout_secs 120.0
conda install mkl pip pytest pytest-cov hypothesis openblas "setuptools>65.5.1"

if [[ "$PYTHON_VERSION" != "3.13" ]]; then
  conda install ecos scs cvxopt proxsuite daqp
  python -m pip install coptpy==7.1.7 gurobipy piqp clarabel osqp
else
  # only install the essential solvers for Python 3.13.
  conda install ecos
  python -m pip install clarabel osqp
fi

if [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.20.
  # Given numpy 1.20, the earliest version of scipy we can use is 1.6.
  conda install scipy=1.6 numpy=1.20
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
  # The earliest version of numpy that works is 1.21.
  # Given numpy 1.21, the earliest version of scipy we can use is 1.7.
  conda install scipy=1.7 numpy=1.21
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
  # The earliest version of numpy that works is 1.23.4.
  # Given numpy 1.23.4, the earliest version of scipy we can use is 1.9.3.
  conda install scipy=1.9.3 numpy=1.23.4
else
  # This case is for Python 3.12 and 3.13.
  # The earliest version of numpy that works is 1.26.4
  # Given numpy 1.26.4, the earliest version of scipy we can use is 1.11.3.
  conda install scipy=1.11.3 numpy=1.26.4
fi

if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  python -m pip install cplex diffcp "ortools>=9.7,<9.10"
fi

if [[ "$RUNNER_OS" == "Windows" ]]; then
  # SDPA with OpenBLAS backend does not pass LP5 on Windows
  python -m pip install sdpa-multiprecision
else
  # cylp has no wheels for Windows
  conda install pyscipopt
  python -m pip install cylp
fi

if [[ "$PYTHON_VERSION" != "3.9" ]] && [[ "$PYTHON_VERSION" != "3.12" ]]; then
  # SDPA didn't pass LP5 on Ubuntu for Python 3.9 and 3.12
  python -m pip install sdpa-python
fi

if [[ "$PYTHON_VERSION" == "3.11" ]] && [[ "$RUNNER_OS" != "macOS" ]]; then
  python -m pip install xpress
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]]; then
    python -m pip install mosek
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  conda install -c conda-forge openmp
fi
