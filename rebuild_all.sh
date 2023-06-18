# Assume we're in the same directory as setup.py

cd cvxpy/cvxcore
swig -py3 -Isrc -c++ -python python/cvxcore.i
cd ../../cvxpy/utilities/cpp/sparsecholesky_swig
swig -py3 -Isrc -c++ -Wextra -python sparsecholesky_swig.i
cd ../../../../
pip install -e .
