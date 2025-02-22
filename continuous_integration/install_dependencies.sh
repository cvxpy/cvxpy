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

conda install scs
python -m pip install clarabel osqp

python -m pip install cvxpygen

# Install newest stable versions for Python 3.13.
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
  conda install scipy numpy
else
  conda install scipy=1.13.0 numpy=1.26.4
fi

if [[ "$USE_OPENMP" == "True" ]]; then
  conda install -c conda-forge openmp
fi
