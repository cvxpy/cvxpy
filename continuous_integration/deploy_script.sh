#!/bin/bash

# Consider deploying to PyPI
if [ $DEPLOY_SOURCE = true ] && [ $TRAVIS_OS_NAME = osx ]; then
    source activate testenv
    conda install --yes requests
    REMOTE_VERSION=`python continuous_integration/versiongetter.py`
    LOCAL_VERSION=`python -c "import cvxpy; print(cvxpy.__version__)"`
    if [ $REMOTE_VERSION -ne $LOCAL_VERSION ]; then
        # Assume the local version is ahead of remote version
        conda install --yes twine
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_TEST_USER -p $PYPI_TEST_PASSWORD
    fi
fi
