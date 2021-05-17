#!/bin/bash

source activate testenv
conda config --add channels conda-forge
conda install --yes requests twine
pip install readme_renderer

# We chose a somewhat arbitrary build configuration (a specially marked OSX configuration)
# to be the designated uploader of source distributions.
if [ $DEPLOY_PYPI_SOURCE == "True" ] && [ $RUNNER_OS == "macOS" ]; then
    # consider deploying to PyPI
    cd continuous_integration
    UPDATE_PYPI_SOURCE=`python -c "import versiongetter as vg; print(vg.update_pypi_source('$PYPI_API_ENDPOINT'))"`
    cd ..
    if [ $UPDATE_PYPI_SOURCE == "True" ]; then
        # assume that local version is ahead of remote version, and update sdist
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
        rm -rf dist
    fi
fi

cd continuous_integration
UPDATE_PYPI_WHEEL=`python -c "import versiongetter as vg; print(vg.update_pypi_wheel('$PYTHON_VERSION','$RUNNER_OS','$PYPI_API_ENDPOINT'))"`
cd ..
if [ $UPDATE_PYPI_WHEEL == "True" ]; then
    python setup.py bdist_wheel
    twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_USER -p $PYPI_PASSWORD
fi
