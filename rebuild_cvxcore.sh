# Assume we're in the same directory as setup.py

cd cvxpy/cvxcore
swig -py3 -Isrc -c++ -python python/cvxcore.i
swig -py3 -Isrc -c++ -python python/sparsecholesky.i
cd ..
cd ..
pip install -e .
