#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

sudo apt-get update -qq
if [[ "$INSTALL_ATLAS" == "true" ]]; then
    #sudo apt-get install -qq libatlas-base-dev
    sudo apt-get install -qq libatlas-base-dev gfortran
    export ATLAS="/usr/lib/atlas-base/libatlas.so"
fi

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-3.7.3-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv

    if [[ "$INSTALL_GLPK" == "true" ]]; then
        # Install GLPK.
        wget http://ftp.gnu.org/gnu/glpk/glpk-4.55.tar.gz
        tar -zxvf glpk-4.55.tar.gz
        cd glpk-4.55
        sudo ./configure
        sudo make
        sudo make install
        cd ..
        # Install CVXOPT with GLPK bindings.
        CVXOPT_BUILD_GLPK=1
        CVXOPT_GLPK_LIB_DIR=/home/travis/glpk-4.55/lib
        CVXOPT_GLPK_INC_DIR=/home/travis/glpk-4.55/include
        pip install cvxopt
    fi

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get update -qq
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-pip python-scipy python-numpy
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi