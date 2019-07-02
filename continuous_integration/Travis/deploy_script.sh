#!/bin/bash

source activate testenv
conda config --add channels cvxgrp
conda config --add channels conda-forge
conda config --add channels oxfordcontrol
conda install --yes requests twine readme_renderer

# We chose a somewhat arbitrary build configuration (a specially marked OSX configuration)
# to be the designated uploader of source distributions.
if [ $DEPLOY_PYPI_SOURCE == "True" ] && [ $TRAVIS_OS_NAME == "osx" ]; then
    # consider deploying to PyPI
    cd continuous_integration
    UPDATE_PYPI_SOURCE=`python -c "import versiongetter as vg; print(vg.update_pypi_source('$PYPI_API_ENDPOINT'))"`
    cd ..
    if [ $UPDATE_PYPI_SOURCE == True ]; then
        # assume that local version is ahead of remote version, and update sdist
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
        rm -rf dist
    fi
fi

cd continuous_integration
UPDATE_PYPI_WHEEL=`python -c "import versiongetter as vg; print(vg.update_pypi_wheel('$PYTHON_VERSION','$TRAVIS_OS_NAME','$PYPI_API_ENDPOINT'))"`
cd ..
if [ $UPDATE_PYPI_WHEEL == "True" ]; then
    python setup.py bdist_wheel
    twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
fi

cd continuous_integration
UPDATE_CONDA=`python -c "import versiongetter as vg; print(vg.update_conda('$PYTHON_VERSION','$TRAVIS_OS_NAME'))"`
cd ..
if [ $UPDATE_CONDA == "True" ]; then
    conda install --yes conda-build
    conda install --yes anaconda-client
    conda config --set anaconda_upload yes
    conda build conda-recipe --token=$CONDA_UPLOAD_TOKEN --user=$CONDA_USER --python=$PYTHON_VERSION
fi
