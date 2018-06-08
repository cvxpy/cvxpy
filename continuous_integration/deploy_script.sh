#!/bin/bash

conda config --add channels cvxgrp
conda config --add channels conda-forge
source activate testenv
conda install --yes requests
cd continuous_integration
REMOTE_PYPI_VERSION=`python -c "import versiongetter; print(versiongetter.pypi_version())"`
REMOTE_CONDA_VERSION=`python -c "import versiongetter; print(versiongetter.conda_version('$PYTHON_VERSION',
'$TRAVIS_OS_NAME'))"`
cd ..
LOCAL_VERSION=`python -c "import cvxpy; print(cvxpy.__version__)"`

if [ $REMOTE_PYPI_VERSION != $LOCAL_VERSION ]; then
    # Consider deploying to PyPI
    if [ $DEPLOY_PYPI = true ] && [ $TRAVIS_OS_NAME = osx ]; then
        # Assume the local version is ahead of remote version
        conda install --yes twine
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
    fi
fi

if [ $REMOTE_CONDA_VERSION != $LOCAL_VERSION ]; then
    # Deploy for conda
    conda install --yes conda-build
    conda install --yes anaconda-client
    conda config --set anaconda_upload yes
    conda build --token=$CONDA_UPLOAD_TOKEN --user=$CONDA_USER --python=$PYTHON_VERSION .
fi
