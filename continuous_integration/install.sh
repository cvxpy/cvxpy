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
        conda install -c conda-forge --yes lapack
        conda install -c cvxgrp --yes ecos scs multiprocess
        conda install -c anaconda --yes flake8

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
        CVXOPT_BUILD_GLPK=1
        CVXOPT_GLPK_LIB_DIR=/usr/local/lib
        CVXOPT_GLPK_INC_DIR=/usr/local/include
        conda install -c cvxgrp --yes cvxopt

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

        # Install cyLP -> which is needed for CBC-interface
        git clone -b py3 https://github.com/jjhelmus/CyLP.git  # use custom-branch because of py3
        cd CyLP
        python setup.py install
        cd ..

        cd "$oldpath"
    elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
             -O miniconda.sh
        chmod +x miniconda.sh && ./miniconda.sh -b
        export PATH=/Users/travis/miniconda3/bin:$PATH
        # Configure the conda environment and put it in the path using the
        # provided versions
        conda create -n testenv --yes python=$PYTHON_VERSION mkl pip nose \
              numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
        source activate testenv
        conda install -c cvxgrp --yes ecos scs multiprocess
        conda install -c anaconda --yes flake8
    fi


elif [[ "$DISTRIB" == "ubuntu" ]]; then
    sudo apt-get update -qq
    # Use standard ubuntu packages in their default version
    sudo apt-get install -qq python-pip python-scipy python-numpy
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi
