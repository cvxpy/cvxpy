#!/bin/bash
# This script is meant to be called by the "test" step defined in
# build.yml. The behavior of the script is controlled by environment
# variables defined in the build.yml in .github/workflows/.

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ $USE_OPENMP == "True" ] && [ $RUNNER_OS == "Linux" ]; then
    CFLAGS="-fopenmp" LDFLAGS="-lgomp" python setup.py install
    export OMP_NUM_THREADS=4
elif [ $RUNNER_OS == "Linux"] && [[ "$PYTHON_VERSION" == "3.8" ]]; then
    # Needed for the 1.3 build to pass.
    python setup.py install
else
    python -m pip list
    python -m pip install .
fi

python -c "import cvxpy; print(cvxpy.installed_solvers())"

if [[ "$SINGLE_ACTION_CONFIG" == "True" ]]; then
    pytest cvxpy/tests --cov=cvxpy --cov-report xml:coverage.xml
else
    pytest cvxpy/tests
fi

pytest cvxpy/performance_tests
