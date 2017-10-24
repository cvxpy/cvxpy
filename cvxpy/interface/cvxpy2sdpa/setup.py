#!/usr/bin/env python
"""sdpapy.py
cvxpy2sdpa(SDPA CVXPY Interface)

October 2017, Miguel Paredes
"""
from distutils.core import setup, Extension


SDPA_DIR = '/home/miguel/Documents/sdpa/sdpa-7.3.8'
SDPA_LIB = SDPA_DIR
SDPA_INCLUDE = SDPA_DIR

MUMPS_DIR = SDPA_DIR+'/mumps/build'
MUMPS_LIB = MUMPS_DIR + '/lib'
MUMPS_LIBSEQ = MUMPS_DIR + '/libseq'
MUMPS_INCLUDE = MUMPS_DIR + '/include'

# Default path

LAPACK_DIR = '/usr/lib/'

LAPACK_NAME = 'lapack'

BLAS_DIR = '/usr/lib/'

BLAS_NAME = 'blas'

ext_sdpacall = Extension('cvxpy2sdpa.sdpacall.sdpa',
                         ['cvxpy2sdpa/sdpacall/cmodule/sdpamodule.cpp'],
                         include_dirs=[SDPA_INCLUDE, MUMPS_INCLUDE],
                         library_dirs=[SDPA_LIB, MUMPS_LIB,MUMPS_LIBSEQ ,
                                       LAPACK_DIR, BLAS_DIR],
                         libraries=['sdpa', 'dmumps', 'mumps_common', 'pord',
                                    'mpiseq', LAPACK_NAME, BLAS_NAME]
                         )

setup(name='cvxpy2sdpa',
      version='0.1',
      description='SDPA CVXPY Interface',
      author='Miguel Paredes Quinones',
      author_email='miguel.paredes.q@gmail.com',
      packages=['cvxpy2sdpa',
                'cvxpy2sdpa.sdpacall'],
      ext_modules=[ext_sdpacall],
      )
