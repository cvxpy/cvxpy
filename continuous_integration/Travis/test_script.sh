#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ $USE_OPENMP == "True" ] && [ $TRAVIS_OS_NAME == "linux" ]; then
    CFLAGS="-fopenmp" LDFLAGS="-lgomp" python setup.py install
    export OMP_NUM_THREADS=4
else
    python setup.py install
fi

python -c "import cvxpy; print(cvxpy.installed_solvers())"
python $(dirname ${BASH_SOURCE[0]})/../osqp_version.py

if [[ "$COVERAGE" == "true" ]]; then
    export WITH_COVERAGE="--with-coverage"
else
    export WITH_COVERAGE=""
fi

pytest $WITH_COVERAGE cvxpy/tests
pytest $WITH_COVERAGE cvxpy/performance_tests
