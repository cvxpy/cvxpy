#!/usr/bin/env python
"""sdpapy.py
cvxpy2sdpa(SDPA CVXPY Interface)

October 2017, Miguel Paredes
"""

__all__ = ['solve']

from .param import param
from .sdpacall import sdpacall
from scipy import sparse
from numpy import matrix
import time

def solve(mDIM,nBLOCK,bLOCKsTRUCT,bLOCKsTYPE,c,F,option=None):

    timeinfo = dict()
    timeinfo['total'] = time.time()
    # --------------------------------------------------
    # Set parameter
    # --------------------------------------------------
    option = param(option)
    if len(option['print']) != 0 and option['print'] != 'no':
        print('---------- SDPA Start ----------')
    # --------------------------------------------------
    # Check validity
    # --------------------------------------------------

    if nBLOCK != len(bLOCKsTRUCT) or nBLOCK != len(bLOCKsTYPE):
        raise TypeError('cvxpy2sdpa.solve(): number of block does not correspond to blockstruct or blockstype')
        return None

    if not isinstance(c, matrix) and not sparse.issparse(c):
        raise TypeError('cvxpy2sdpa.solve(): c must be a matrix or a sparse matrix.')

    c = sparse.csc_matrix(c)

    for m in range(mDIM+1):
        for block in range(nBLOCK) :
            if not isinstance(F[m][block], matrix)  and not sparse.issparse(F[m][block]) and  not F[m][block] == []:
                raise TypeError('cvxpy2sdpa.solve(): element in F (m = '+str(m)+' , block = '+str(block)+') must be a matrix, must be a sparse matrix or empty array.')
            elif F[m][block] != []:
                if bLOCKsTYPE[block] == 0:
                    if F[m][block].shape[1] != 1 :
                        raise TypeError('cvxpy2sdpa.solve(): element in F (m = ' + str(m) + ' , block = ' + str(block) + ') must be a matrix with colum dimension 1.')

    # Solve by SDPA
    # --------------------------------------------------
    timeinfo['sdpa'] = time.time()
    objVal, x, X, Y, sdpainfo = sdpacall.solve_sdpa(mDIM,nBLOCK,bLOCKsTRUCT,bLOCKsTYPE,c,F,option)
    timeinfo['sdpa'] = time.time() - timeinfo['sdpa']
    if len(option['print']) != 0 and option['print'] != 'no':
        print("     solveTime = %f" % timeinfo['sdpa'])
        print("     totalTime = %f" % timeinfo['total'])
        print('---------- SDPA End ----------')

    return objVal, x, X, Y, sdpainfo, timeinfo