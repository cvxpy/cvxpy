#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # Use miniconda
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
        sudo apt-get update -qq
        sudo apt-get install -qq gfortran libgfortran3
        LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgfortran.so.3
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
           -O miniconda.sh
        chmod +x miniconda.sh && ./miniconda.sh -b
        export PATH=/home/travis/miniconda3/bin:$PATH
        conda update --yes conda
        # Configure the conda environment and put it in the path using the
        # provided versions
        conda create -n testenv --yes python=$PYTHON_VERSION mkl pip nose \
                numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
        source activate testenv
        PIN_FILE=/home/travis/miniconda3/envs/testenv/conda-meta/pinned
        touch $PIN_FILE
        echo "python=$PYTHON_VERSION" >> $PIN_FILE
        conda install -c conda-forge --yes lapack
        conda install -c conda-forge --yes ecos scs multiprocess
        conda install -c default --yes flake8

        # Install GLPK.
        if [[ "$CVXOPT" == "true" ]]; then
            pip install cvxopt
        fi

    elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
             -O miniconda.sh
        chmod +x miniconda.sh && ./miniconda.sh -b
        export PATH=/Users/travis/miniconda3/bin:$PATH
        conda update --yes conda
        # Configure the conda environment and put it in the path using the
        # provided versions
        conda create -n testenv --yes python=$PYTHON_VERSION mkl pip nose \
              numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
        source activate testenv
        PIN_FILE=/Users/travis/miniconda3/envs/testenv/conda-meta/pinned
        touch $PIN_FILE
        echo "python=$PYTHON_VERSION.*" >> $PIN_FILE
        conda install -c conda-forge --yes ecos scs multiprocess
        conda install -c default --yes flake8=3.5.0
    fi


elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get update -qq
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-pip python-scipy python-numpy
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
