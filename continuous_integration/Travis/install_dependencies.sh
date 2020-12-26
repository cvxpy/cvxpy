#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -qq gfortran libgfortran3
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    PIN_FILE=/home/travis/miniconda3/envs/testenv/conda-meta/pinned
    touch $PIN_FILE
    echo "python=$PYTHON_VERSION" >> $PIN_FILE
    export PATH=/home/travis/miniconda3/bin:$PATH
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
         -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    PIN_FILE=/Users/travis/miniconda3/envs/testenv/conda-meta/pinned
    touch $PIN_FILE
    echo "python=$PYTHON_VERSION.*" >> $PIN_FILE
    export PATH=/Users/travis/miniconda3/bin:$PATH
fi

conda update --yes conda
conda config --add channels conda-forge
conda create -n testenv --yes python=$PYTHON_VERSION mkl pip pytest \
      numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
source activate testenv
conda install --yes lapack ecos scs
conda install -c anaconda --yes flake8
#pip install osqp # not available for up-to-date python, as of Dec 26, 2020.
pip install diffcp
pip install cvxopt

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
