#!/bin/bash

set -e

conda config --set remote_connect_timeout_secs 30.0
conda config --set remote_max_retries 10
conda config --set remote_backoff_factor 2
conda config --set remote_read_timeout_secs 120.0
conda install pip

python -m pip install ecos scs proxsuite daqp gurobipy piqp clarabel osqp highspy qoco qpalm xpress

# Skip installing mpax as it causes test_qp_solvers.py to hang when running on macos
# and it fails StandardTestLPs.test_lp_6() and StandardTestLPs.test_lp_2() on ubuntu and windows

# if [[ "$RUNNER_OS" != "macOS" ]] || [[ $(uname -m) != "x86_64" ]]; then
#   python -m pip install mpax
# fi

if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  python -m pip install "ortools>=9.7,<9.15"
fi

if [[ "$RUNNER_OS" == "Windows" ]]; then
  # SDPA with OpenBLAS backend does not pass LP5 on Windows
  python -m pip install sdpa-multiprecision
fi

if [[ "$RUNNER_OS" != "Windows" ]]; then
  python -m pip install cvxopt
fi

if [[ "$RUNNER_OS" != "Windows" ]]; then
  # cylp has no wheels for Windows
  python -m pip install cylp
fi

if [[ "$RUNNER_OS" != "Ubuntu" ]]; then
  # SDPA didn't pass LP5 on Ubuntu for Python 3.9 and 3.12
  python -m pip install sdpa-python
fi

if [[ "$RUNNER_OS" != "macOS" ]]; then
  python -m pip install coptpy==7.1.7 cplex
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]]; then
  python -m pip install mosek
fi

# Only install KNITRO if license is available (secret is not copied to forks)
# KNITRO on macOS is only available for arch64
if [[ -n "$KNITRO_LICENSE" ]] && ! ([[ "$RUNNER_OS" == "macOS" ]] && [[ $(uname -m) == "x86_64" ]]); then
  python -m pip install knitro
fi

# Install and setup python-julia interface
python -m pip install julia
python -c "import julia; julia.install()"

# Install COSMOPY
python -m pip install git+https://github.com/oxfordcontrol/cosmo-python.git