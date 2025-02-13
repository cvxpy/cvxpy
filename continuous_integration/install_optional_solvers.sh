#!/bin/bash

set -e

if [[ "$PYTHON_VERSION" != "3.13" ]]; then
  pip install ecos scs proxsuite daqp
  python -m pip install gurobipy piqp clarabel osqp highspy
else
  # only install the essential solvers for Python 3.13.
  pip install scs
  python -m pip install clarabel osqp
fi

if [[ "$PYTHON_VERSION" != "3.13" ]] && [[ "$RUNNER_OS" != "macOS" ]]; then
  # coptpy does not have wheels for macos-13, only macos-10, 
  # but the runners are on macos-13, so do not install coptpy for macos.
  python -m pip install coptpy==7.1.7
fi

if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  python -m pip install cplex "ortools>=9.7,<9.12"
fi

if [[ "$RUNNER_OS" != "Windows" ]]; then
  # qoco has no wheels for Windows
  python -m pip install qoco
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
  python -m pip install cylp pyscipopt==5.2.1
fi

if [[ "$PYTHON_VERSION" == "3.12" ]] && [[ "$RUNNER_OS" != "Windows" ]]; then
  # SDPA didn't pass LP5 on Ubuntu for Python 3.9 and 3.12
  python -m pip install sdpa-python
fi

if [[ "$PYTHON_VERSION" == "3.12" ]] && [[ "$RUNNER_OS" != "macOS" ]]; then
  python -m pip install xpress==9.4.3
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]] && [[ "$PYTHON_VERSION" != "3.13" ]]; then
    python -m pip install mosek
fi