import numpy as np

import cvxpy as cp

"""
kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
             [M[1,0] * N   , ..., M[1, end] * N  ]
             ...
             [M[end, 0] * N, ..., M[end, end] * N] 
"""


def random_problem(z_dims, c_dims, left=True, param=True, seed=0):
    Z = cp.Variable(shape=z_dims)
    np.random.seed(seed)
    C_value = np.random.rand(*c_dims).round(decimals=2)
    if param:
        C = cp.Parameter(shape=c_dims)
        C.value = C_value
    else:
        C = cp.Constant(C_value)
    L = np.random.rand(*Z.shape)
    U = L + np.random.rand(*Z.shape)
    U = U.round(decimals=2)
    if left:
        constraints = [cp.kron(U, C) >= cp.kron(Z, C), cp.kron(Z, C) >= cp.kron(L, C)]
    else:
        constraints = [cp.kron(C, U) >= cp.kron(C, Z), cp.kron(C, Z) >= cp.kron(C, L)]
    obj_expr = cp.sum(Z)

    prob = cp.Problem(cp.Minimize(obj_expr), constraints)
    return Z, C, U, prob


if __name__ == '__main__':
    """
    Use this script for debugging kron.
    It will probably help to take screenshots of debugging state variables when
    running under the old functionality and the new functionality.
    You can find a set of screenshots at:
    https://docs.google.com/presentation/d/1ETKlAkz1XrSfikvnF27T-JGYF5niLoA936fOSduvy6U/edit?usp=sharing
    feel free to edit by adding more slides.
    As of Feb 11, the relevant slides start on page 18.
    """
    seed = 0
    z_dims = (2, 2)
    c_dims = (2, 3)

    LEFT = True  # True means we test the new functionality, with a Variable in left argument
    solve = False  # I haven't found much value in setting this to True
    param = True  # tests pass when param=False. But maybe worth looking at ...
    #   (param, LEFT) = (False, False) shows what kron(const, var) canonicalizes to w/o parameters
    #   (param, LEFT) = (False, True) shows what kron(var, const) canonicalizes to w/o parameters

    Z_t, C_t, U_t, prob = random_problem(z_dims, c_dims, left=LEFT, param=True, seed=0)
    if solve:
        prob.solve(solver='ECOS')
    else:
        data_t = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
        G = data_t['G'].A
        h = data_t['h']
        print('\nDimensions')
        print(f'\tz_dims = {z_dims}')
        print(f'\tc_dims = {c_dims}')
        print(f'\nData for param=True and LEFT={LEFT}')
        print(f'\tnnz(G) = {np.count_nonzero(G)}')
        print(f'\tnnz(h) = {np.count_nonzero(h)}')
        print()

