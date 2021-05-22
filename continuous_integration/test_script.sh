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
else
    python setup.py install
fi

python -c "import cvxpy; print(cvxpy.installed_solvers())"
python $(dirname ${BASH_SOURCE[0]})/osqp_version.py

if [[ "$COVERAGE" == "True" ]]; then
    export WITH_COVERAGE="--with-coverage"
else
    export WITH_COVERAGE=""
fi

pytest $WITH_COVERAGE cvxpy/tests
pytest $WITH_COVERAGE cvxpy/performance_tests
