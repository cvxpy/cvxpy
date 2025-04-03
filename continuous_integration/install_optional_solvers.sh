#!/bin/bash

set -e

conda config --set remote_connect_timeout_secs 30.0
conda config --set remote_max_retries 10
conda config --set remote_backoff_factor 2
conda config --set remote_read_timeout_secs 120.0
conda install pip

if [[ "$PYTHON_VERSION" != "3.13" ]]; then
  python -m pip install ecos scs proxsuite daqp gurobipy piqp clarabel osqp highspy qoco
else
  # only install the essential solvers for Python 3.13.
  python -m pip install scs clarabel osqp
fi

if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  python -m pip install "ortools>=9.7,<9.12"
fi

if [[ "$RUNNER_OS" == "Windows" ]] && [[ "$PYTHON_VERSION" != "3.13" ]]; then
  # SDPA with OpenBLAS backend does not pass LP5 on Windows
  python -m pip install sdpa-multiprecision
fi

if [[ "$RUNNER_OS" != "Windows" ]] && [[ "$PYTHON_VERSION" != "3.13" ]]; then
  python -m pip install cvxopt
fi

if [[ "$PYTHON_VERSION" == "3.12" ]] && [[ "$RUNNER_OS" != "Windows" ]]; then
  # cylp has no wheels for Windows
  python -m pip install cylp
fi

if [[ "$PYTHON_VERSION" == "3.12" ]] && [[ "$RUNNER_OS" != "Ubuntu" ]]; then
  # SDPA didn't pass LP5 on Ubuntu for Python 3.9 and 3.12
  python -m pip install sdpa-python
fi

if [[ "$PYTHON_VERSION" == "3.12" ]] && [[ "$RUNNER_OS" != "macOS" ]]; then
  python -m pip install xpress==9.4.3 coptpy==7.1.7 cplex
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]] && [[ "$PYTHON_VERSION" != "3.13" ]]; then
    python -m pip install mosek
fi