import numpy as np

import cvxpy as cp

"""
kron(M, N) = [M[0,0] * N   , ..., M[0, end] * N  ]
             [M[1,0] * N   , ..., M[1, end] * N  ]
             ...
             [M[end, 0] * N, ..., M[end, end] * N] 
"""


def print_array_indented(mat, indent_level=1):
    tab_str = indent_level * '\t'
    mat_str = str(mat)
    mat_str = tab_str + mat_str
    mat_str = mat_str.replace('\n ', f'\n{tab_str} ')
    print(mat_str)


def random_problem(z_dims, c_dims, var_left, param, seed=0):
    """
    Construct random nonnegative matrices (C, L) of shapes
    (c_dims, z_dims) respectively. Define an optimization
    problem with a matrix variable of shape z_dims:

        min sum(Z)
        s.t.  kron(Z, C) >= kron(L, C)   ---   if var_left is True
              kron(C, Z) >= kron(C, L)   ---   if var_left is False
              Z >= 0

    Regardless of whether var_left is True or False, the optimal
    solution to that problem is Z = L.

    If param is True, then C is defined as a CVXPY Parameter.
    If param is False, then C is a CVXPY Constant.

    A small remark: the constraint that Z >= 0 is redundant.
    It's there because it's easier to set break points that distinguish
    objective canonicalization and constraint canonicalization
    when there's more than one constraint.
    """
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
    _obj_expr = cp.sum(_Z)
    _prob = cp.Problem(cp.Minimize(_obj_expr), _constraints)
    return _Z, _C, _L, _prob


def run_example(z_dims, c_dims, var_left=True, param=True, solve=True, seed=0):
    print('\nContext for this run ...')
    print(f'\tshape of Variable  Z: {z_dims}')
    c_desc = 'Parameter' if param else 'Constant '
    print(f'\tshape of {c_desc} C: {c_dims}')
    side = 'LEFT' if var_left else 'RIGHT'
    when = 'NEW' if var_left else 'OLD'
    print(f'\tZ is the {side} operand to kron; this is {when} functionality.')
    Z, C, L, prob = random_problem(z_dims, c_dims, var_left, param, seed=seed)

    if solve:
        prob.solve(solver='ECOS')
        print('\nSolving with ECOS ...')
        violations = prob.constraints[0].violation()
        print(f'\tProblem status: {prob.status}')
        print(f'\tZ.value = ...\n')
        print_array_indented(Z.value, indent_level=2)
        print(f'\n\tConstraint violation: {np.max(violations)}')

    data = prob.get_problem_data(solver='ECOS', enforce_dpp=True)[0]
    # ^ Changing to enforce_dpp=False doesn't make a difference
    G = data['G'].A
    h = data['h']
    print('\nThe matrix G in ECOS data "G x <= h" is \n')
    print_array_indented(G, indent_level=1)
    print('\nThe vector h in ECOS data "G x <= h" is \n')
    print_array_indented(h, indent_level=1)
    return Z, C, L, prob, G, h


if __name__ == '__main__':
    """
    Use this script for debugging kron.
    It will probably help to take screenshots of debugging state variables when
    running under the old functionality and the new functionality.
    You can find a set of screenshots at:
    https://docs.google.com/presentation/d/1ETKlAkz1XrSfikvnF27T-JGYF5niLoA936fOSduvy6U/edit?usp=sharing
    feel free to edit by adding more slides.
    """
    seed = 0
    z_dims = (1, 1)
    c_dims = (1, 2)

    solve = True
    var_left = False  # True means we test the new functionality, with a Variable in left argument
    param = True  # tests pass when param=False. But maybe worth looking at ...
    #   (param, var_left) = (False, False) shows what kron(const, var) canonicalizes to w/o parameters
    #   (param, var_left) = (False, True) shows what kron(var, const) canonicalizes to w/o parameters

    Z, C, L, prob, G, h = run_example(z_dims, c_dims, var_left, param, solve, seed)
