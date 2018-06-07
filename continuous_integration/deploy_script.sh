#!/bin/bash

# Consider deploying to PyPI
if [ $DEPLOY_SOURCE = true ] && [ $TRAVIS_OS_NAME = osx ]; then
    REMOTE_VERSION=`python continuous_integration/versiongetter.py`
    LOCAL_VERSION=`python -c "import cvxpy; print(cvxpy.__version__)"`
    if [ $REMOTE_VERSION -ne $LOCAL_VERSION ]; then
        # Assume the local version is ahead of remote version
        source activate testenv
        conda install twine
        python setup.py sdist
        twine upload --repository-url $PYPI_SERVER dist/* -u $PYPI_TEST_USER -p $PYPI_TEST_PASSWORD
    fi
fi
