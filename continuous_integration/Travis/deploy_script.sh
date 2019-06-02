#!/bin/bash

source activate testenv
conda config --add channels cvxgrp
conda config --add channels conda-forge
conda config --add channels oxfordcontrol
conda install --yes requests

# Right now, we only update the source distribution.
#   We chose a somewhat arbitrary build configuration (a specially marked OSX configuration)
#   to be the designated uploader of source distributions.
if [ $DEPLOY_PYPI = true ] && [ $TRAVIS_OS_NAME = osx ]; then
    # consider deploying to PyPI
    cd continuous_integration
    REMOTE_PYPI_VERSION=`python -c "import versiongetter as vg; print(vg.pypi_version('$PYPI_SERVER'))"`
    cd ..
    LOCAL_VERSION=`python -c "import cvxpy; print(cvxpy.__version__)"`
    if [ $REMOTE_PYPI_VERSION != $LOCAL_VERSION ]; then
        # assume that local version is ahead of remote version, and update sdist
        conda install --yes twine
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
    fi
fi

# We always update the conda builds.
cd continuous_integration
UPDATE_CONDA=`python -c "import versiongetter as vg; print(vg.update_conda('$PYTHON_VERSION','$TRAVIS_OS_NAME'))"`
cd ..
if [ $UPDATE_CONDA == True ]; then
    # Deploy for conda
    conda install --yes conda-build
    conda install --yes anaconda-client
    conda config --set anaconda_upload yes
    conda build conda-recipe --token=$CONDA_UPLOAD_TOKEN --user=$CONDA_USER --python=$PYTHON_VERSION
fi
