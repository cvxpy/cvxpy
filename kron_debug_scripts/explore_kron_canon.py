import numpy as np

import cvxpy as cp

"""
kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
             [M[1,0] * N   , ..., M[1, end] * N  ]
             ...
             [M[end, 0] * N, ..., M[end, end] * N] 
"""


def random_problem(z_dims, c_dims, var_left, param, seed=0):
    np.random.seed(seed)
    _C_value = np.random.rand(*c_dims).round(decimals=2)
    if param:
        _C = cp.Parameter(shape=c_dims)
        _C.value = _C_value
    else:
        _C = cp.Constant(_C_value)
    _Z = cp.Variable(shape=z_dims)
    _L = np.random.rand(*z_dims).round(decimals=2)
    if var_left:
        _constraints = [cp.kron(_Z, _C) >= cp.kron(_L, _C), _Z >= 0]
    else:
        _constraints = [cp.kron(_C, _Z) >= cp.kron(_C, _L), _Z >= 0]
    # ^ Only the first constraint matters. We use two constraints because that
    # makes it easier to set conditional breakpoints in canonInterface.py.
    #
    #   Specifically, canonInterface.py:get_problem_matrix is called once for
    #   the objective function and once for the constraint matrix. When it's
    #   called for the objective function the linOps argument is a list of length
    #   one, and when it's called for the constraint matrix the linOps argument
    #   is a list of length equal to the number of constraints.
    #
    _obj_expr = cp.sum(_Z)

    _prob = cp.Problem(cp.Minimize(_obj_expr), _constraints)
    # The optimal solution is Z.value == L.
    return _Z, _C, _L, _prob


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
    c_dims = (1, 2)

    solve = False  # I haven't found much value in setting this to True
    var_left = False  # True means we test the new functionality, with a Variable in left argument
    param = True  # tests pass when param=False. But maybe worth looking at ...
    #   (param, LEFT) = (False, False) shows what kron(const, var) canonicalizes to w/o parameters
    #   (param, LEFT) = (False, True) shows what kron(var, const) canonicalizes to w/o parameters

    Z, C, L, prob = random_problem(z_dims, c_dims, var_left, param, seed=seed)
    if solve:
        prob.solve(solver='ECOS')
    else:
        print('\nDimensions')
        print(f'\tz_dims = {z_dims}')
        print(f'\tc_dims = {c_dims}')
        data = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
        G = data['G'].A
        h = data['h']
        print(f'\nData for param={param} and var_left={var_left}')
        print(f'\tnnz(G) = {np.count_nonzero(G)}')
        print(f'\tnnz(h) = {np.count_nonzero(h)}\n')
        print(G)
        print(h)
        print()

        data = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
        G = data['G'].A
        h = data['h']
        print(f'\nData for param={param} and var_left={var_left}')
        print(f'\tnnz(G) = {np.count_nonzero(G)}')
        print(f'\tnnz(h) = {np.count_nonzero(h)}\n')
        print(G)
        print(h)
        print()

    Z, C, L, prob = random_problem(z_dims, c_dims, var_left, param, seed=seed)
    if solve:
        prob.solve(solver='ECOS')
    else:
        print('\nDimensions')
        print(f'\tz_dims = {z_dims}')
        print(f'\tc_dims = {c_dims}')
        data = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
        G = data['G'].A
        h = data['h']
        print(f'\nData for param={param} and var_left={var_left}')
        print(f'\tnnz(G) = {np.count_nonzero(G)}')
        print(f'\tnnz(h) = {np.count_nonzero(h)}\n')
        print(G)
        print(h)
        print()