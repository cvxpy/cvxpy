#!/usr/bin/env python
"""sdpacall.py

A module of sdpapy
call sdpa

October 2017, Miguel Paredes
"""

__all__ = ['solve_sdpa']

from .sdpa import *
from cvxpy2sdpa.matdata import MatData
from scipy import sparse
import numpy as np

def solve_sdpa(mDIM,nBLOCK,bLOCKsTRUCT,bLOCKsTYPE,c,F, option):
    """Solve SDP with sdpa
    Args:
        mDIM : number of constraints
        nBLOCK: number of matrix blocks
        bLOCKsTRUCT: dimension of blocks
        bLOCKsTYPE: tipo de bloco, semidefite or linear positive
        c: right hand constraint vector
        F:  contraint plus objective function matrizes
        option : dictionary with options

    Returns:
        objVal : primal and dual objective function
        x: primal variable vector
        X: primal variable matrix
        Y: dual variable matrix

    Primal :
            Min c^Tx
            X_{j} = \sum{i=1,..,mDIM} F_{j,i} x_{j,i} +  F_{j,0}  >=0 \forall j = 1,...,nBLOCK

    Dual: Max \sum_{j=1,...,nBLOCK} F_{j,0}*Y_{j}
            \sum_{j=1,...,nBLOCK} F_{j,i}*Y_{j}  = c_{i} \forall i=1,..,mDIM
            Y_{j} >=0 \forall i=1,..,mDIM
    """
    data_c = MatData(c)

    data_F = [ ]

    for m in range(mDIM+1):
        data_F.append([])
        for block in range(nBLOCK) :
            if not sparse.issparse(F[m][block]) :
                F[m][block] = sparse.csc_matrix(F[m][block])

            data_F[m].append(MatData(F[m][block]))

    objVal,x,X,Y,info = sdpasolver(bLOCKsTRUCT,bLOCKsTYPE,data_c,data_F,option)

    for block in range(nBLOCK) :
        X[block] = np.array(X[block])
        Y[block] = np.array(Y[block])

    return objVal,np.array(x),X,Y, info