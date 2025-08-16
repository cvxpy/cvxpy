#!/bin/bash

set -e
uv venv
if [[ "$RUNNER_OS" == "Windows" ]]; then
  . .venv/Scripts/activate
else
  . .venv/bin/activate
fi

uv pip install ecos scs proxsuite daqp gurobipy piqp clarabel osqp highspy qoco 

if [[ "$RUNNER_OS" != "macOS" ]] || [[ $(uname -m) != "x86_64" ]]; then
  uv pip install jax mpax
fi

if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  uv pip install "ortools>=9.7,<9.15"
fi

if [[ "$RUNNER_OS" == "Windows" ]]; then
  # SDPA with OpenBLAS backend does not pass LP5 on Windows
  uv pip install sdpa-multiprecision
fi

if [[ "$RUNNER_OS" != "Windows" ]]; then
  uv pip install cvxopt
fi

if [[ "$RUNNER_OS" != "Windows" ]]; then
  # cylp has no wheels for Windows
  uv pip install cylp
fi

if [[ "$RUNNER_OS" != "Ubuntu" ]]; then
  # SDPA didn't pass LP5 on Ubuntu for Python 3.9 and 3.12
  uv pip install sdpa-python
fi

if [[ "$RUNNER_OS" != "macOS" ]]; then
  uv pip install xpress coptpy==7.1.7 cplex
fi

# Only install Mosek if license is available (secret is not copied to forks)
if [[ -n "$MOSEK_CI_BASE64" ]]; then
  uv pip install mosek
fi

# Only install KNITRO if license is available (secret is not copied to forks)
# KNITRO on macOS is only available for arch64
if [[ -n "$KNITRO_LICENSE" ]] && ! ([[ "$RUNNER_OS" == "macOS" ]] && [[ $(uname -m) == "x86_64" ]]); then
  uv pip install knitro
fi

# Install and setup python-julia interface
uv pip install julia
python -c "import julia; julia.install()"

# Install COSMOPY
uv pip install git+https://github.com/oxfordcontrol/cosmo-python.git
