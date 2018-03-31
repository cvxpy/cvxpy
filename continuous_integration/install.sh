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
    wget http://repo.continuum.io/miniconda/Miniconda-3.9.1-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    if [[ "$INSTALL_GLPK" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION nomkl pip nose \
              numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    else
        conda create -n testenv --yes python=$PYTHON_VERSION mkl pip nose \
              numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    fi
    source activate testenv
    conda install -c cvxgrp --yes ecos multiprocess
    pip install flake8

    if [[ "$INSTALL_GLPK" == "true" ]]; then
        # Install GLPK.
        wget http://ftp.gnu.org/gnu/glpk/glpk-4.60.tar.gz
        tar -zxvf glpk-4.60.tar.gz
        cd glpk-4.60
        sudo ./configure
        sudo make
        sudo make check
        sudo make install
        sudo ldconfig
        cd ..
        # Install CVXOPT with GLPK bindings.
        CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR=/usr/local/lib CVXOPT_GLPK_INC_DIR=/usr/local/include pip install cvxopt

        # Install CBC
        oldpath="$PWD"
        cd /home/travis
        wget http://www.coin-or.org/download/source/Cbc/Cbc-2.9.7.tgz
        tar -zxvf Cbc-2.9.7.tgz
        cd Cbc-2.9.7
        sudo ./configure
        sudo make
        sudo make install
        cd ..

        export COIN_INSTALL_DIR=/home/travis/Cbc-2.9.7
        export LD_LIBRARY_PATH=/home/travis/Cbc-2.9.7/lib:$LD_LIBRARY_PATH

        # sudo apt-get install coinor-libcbc-dev coinor-libcbc0 coinor-libcbc-doc

        # Install cyLP -> which is needed for CBC-interface
        git clone -b py3 https://github.com/jjhelmus/CyLP.git  # use custom-branch because of py3
        cd CyLP
        python setup.py install
        cd ..

        cd "$oldpath"
    fi

    # if [[ "$INSTALL_MKL" == "true" ]]; then
    #     # Make sure that MKL is used
    #     conda install --yes mkl
    # else
    #     # Make sure that MKL is not used
    #     conda remove --yes --features mkl || echo "MKL not installed"
    # fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get update -qq
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-pip python-scipy python-numpy
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
