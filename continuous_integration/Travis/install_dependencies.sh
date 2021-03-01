#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -qq gfortran libgfortran3
    # LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda3/bin:$PATH
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
         -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/Users/travis/miniconda3/bin:$PATH
fi

conda update --yes conda
conda config --add channels anaconda
conda config --add channels conda-forge
# ^ Adding conda-forge second has the effect of giving it higher priority.

if [[ "$PYTHON_VERSION" == "3.7" ]]; then
  conda create -n testenv --yes python=3.7 mkl pip pytest \
   scipy=1.1 numpy=1.15 lapack ecos scs osqp flake8 cvxopt
  source activate testenv
elif [[ "$PYTHON_VERSION" == "3.8" ]]; then
  # There is a config that works with numpy 1.14, but not 1.15!
  # So we fix things at 1.16.
  # Assuming we use numpy 1.16, the earliest version of scipy we can use is 1.3.
  conda create -n testenv --yes python=3.8 mkl pip pytest \
   scipy=1.3 numpy=1.16 lapack ecos scs osqp flake8 cvxopt
  source activate testenv
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
  # The earliest version of numpy that works is 1.19.
  # Given numpy 1.19, the earliest version of scipy we can use is 1.5.
  conda create -n testenv --yes python=3.9 mkl pip pytest \
   scipy=1.5 numpy=1.19 lapack ecos scs flake8 cvxopt
  source activate testenv
  pip install osqp
fi

if [[ "$USE_OPENMP" == "true" ]]; then
    conda install -c conda-forge openmp
fi

pip install diffcp

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
