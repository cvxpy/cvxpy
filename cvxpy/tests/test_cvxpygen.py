"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import glob
import os
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp

try:
    from cvxpygen import cpg
    cpg_installed = True
except ImportError:
    cpg_installed = False


def network_problem():

    # define dimensions
    n, m = 10, 5

    # define variable
    f = cp.Variable(n, name='f')

    # define parameters
    R = cp.Parameter((m, n), name='R')
    c = cp.Parameter(m, nonneg=True, name='c')
    w = cp.Parameter(n, nonneg=True, name='w')
    f_min = cp.Parameter(n, nonneg=True, name='f_min')
    f_max = cp.Parameter(n, nonneg=True, name='f_max')

    # define objective
    objective = cp.Maximize(w @ f)

    # define constraints
    constraints = [R @ f <= c, f_min <= f, f <= f_max]

    # define problem
    return cp.Problem(objective, constraints)


def MPC_problem():

    # define dimensions
    H, n, m = 10, 6, 3

    # define variables
    U = cp.Variable((m, H), name='U')
    X = cp.Variable((n, H + 1), name='X')

    # define parameters
    Psqrt = cp.Parameter((n, n), name='Psqrt', diag=True)
    Qsqrt = cp.Parameter((n, n), name='Qsqrt', diag=True)
    Rsqrt = cp.Parameter((m, m), name='Rsqrt', diag=True)
    nonzeros_A = [(i, i) for i in range(n)] + [(i, 3+i) for i in range(n // 2)]
    A = cp.Parameter((n, n), name='A', sparsity=tuple(zip(*nonzeros_A)))
    nonzeros_B = [(3+i, i) for i in range(n // 2)]
    B = cp.Parameter((n, m), name='B', sparsity=tuple(zip(*nonzeros_B)))
    x_init = cp.Parameter(n, name='x_init')

    # define objective
    objective = cp.Minimize(
        cp.sum_squares(Psqrt @ X[:, H - 1]) +
        cp.sum_squares(Qsqrt @ X[:, :H]) +
        cp.sum_squares(Rsqrt @ U)+1
    )

    # define constraints
    constraints = [X[:, 1:] == A @ X[:, :H] + B @ U,
                   cp.abs(U) <= 1,
                   X[:, 0] == x_init]

    # define problem
    return cp.Problem(objective, constraints)


def ADP_problem():

    # define dimensions
    n, m = 6, 3

    # define variables
    u = cp.Variable(m, name='u')

    # define parameters
    Rsqrt = cp.Parameter((m, m), name='Rsqrt', diag=True)
    f = cp.Parameter(n, name='f')
    G = cp.Parameter((n, m), name='G')

    # define objective
    objective = cp.Minimize(cp.sum_squares(f + G @ u) + cp.sum_squares(Rsqrt @ u)+98)

    # define constraints
    constraints = [cp.norm(u, 2) <= 1]

    # define problem
    return cp.Problem(objective, constraints)


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'network':

        n, m = 10, 5
        prob.param_dict['R'].value = np.round(np.random.rand(m, n))
        prob.param_dict['c'].value = n * (0.1 + 0.1 * np.random.rand(m))
        prob.param_dict['w'].value = np.random.rand(n)
        prob.param_dict['f_min'].value = np.zeros(n)
        prob.param_dict['f_max'].value = np.ones(n)

    elif name == 'MPC':

        # continuous-time dynmaics
        A_cont = np.concatenate((np.array([[0, 0, 0, 1, 0, 0],
                                           [0, 0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 0, 1]]),
                                           np.zeros((3, 6))), axis=0)
        mass = 1
        B_cont = np.concatenate((np.zeros((3, 3)),
                                 (1 / mass) * np.diag(np.ones(3))), axis=0)

        # discrete-time dynamics
        td = 0.1

        prob.param_dict['A'].value_sparse = sp.coo_array(np.eye(6) + td * A_cont)
        prob.param_dict['B'].value_sparse = sp.coo_array(td * B_cont)

        prob.param_dict['A'].value_sparse = sp.coo_array(np.eye(6) + td * A_cont)
        prob.param_dict['B'].value_sparse = sp.coo_array(td * B_cont)
        prob.param_dict['Psqrt'].value = np.eye(6)
        prob.param_dict['Qsqrt'].value = np.eye(6)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['x_init'].value = -2*np.ones(6) + 4*np.random.rand(6)

    elif name == 'ADP':

        def dynamics(x):
            # continuous-time dynmaics
            A_cont = np.array([[0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, -x[3], 0, 0],
                               [0, 0, 0, 0, -x[4], 0],
                               [0, 0, 0, 0, 0, -x[5]]])
            mass = 1
            B_cont = np.concatenate((np.zeros((3, 3)),
                                     (1 / mass) * np.diag(x[3:])), axis=0)
            # discrete-time dynamics
            td = 0.1
            return np.eye(6) + td * A_cont, td * B_cont

        state = -2*np.ones(6) + 4*np.random.rand(6)
        Psqrt = np.eye(6)
        A, B = dynamics(state)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['f'].value = np.matmul(Psqrt, np.matmul(A, state))
        prob.param_dict['G'].value = np.matmul(Psqrt, B)

    return prob


def get_primal_vec(prob, name):
    if name == 'network':
        return prob.var_dict['f'].value
    if name == 'MPC':
        return np.concatenate(
            (prob.var_dict['U'].value.flatten(), prob.var_dict['X'].value.flatten())
        )
    if name == 'ADP':
        return prob.var_dict['u'].value
    return None


def get_dual_vec(prob):
    dual_values = []
    for constr in prob.constraints:
        if constr.args[0].size == 1:
            dual_values.append(np.atleast_1d(constr.dual_value).flatten())
        else:
            dual_values.append(constr.dual_value.flatten())
    return np.concatenate(dual_values)


def nan_to_inf(val):
    if np.isnan(val):
        return np.inf
    return val

def check(prob, solver, name, func_get_primal_vec, **extra_settings):

    if solver == 'OSQP':
        val_py = prob.solve(
            solver='OSQP', eps_abs=1e-3, eps_rel=1e-3,
            eps_prim_inf=1e-4, eps_dual_inf=1e-4, delta=1e-6,
            max_iter=4000, polish=False, adaptive_rho_interval=int(1e6),
            warm_start=False, **extra_settings
        )
    elif solver == 'SCS':
        val_py = prob.solve(solver='SCS', warm_start=False, verbose=False, **extra_settings)
    else:
        val_py = prob.solve(solver=solver, **extra_settings)
    prim_py = func_get_primal_vec(prob, name)
    dual_py = get_dual_vec(prob)
    if solver == 'OSQP':
        val_cg = prob.solve(method='CPG', warm_start=False, **extra_settings)
    elif solver == 'SCS':
        val_cg = prob.solve(method='CPG', warm_start=False, verbose=False, **extra_settings)
    else:
        val_cg = prob.solve(method='CPG', **extra_settings)
    prim_cg = func_get_primal_vec(prob, name)
    dual_cg = get_dual_vec(prob)
    prim_py_norm = np.linalg.norm(prim_py, 2)
    dual_py_norm = np.linalg.norm(dual_py, 2)

    return (
        nan_to_inf(val_py), prim_py, dual_py, nan_to_inf(val_cg),
        prim_cg, dual_cg, prim_py_norm, dual_py_norm
    )


N_RAND = 1

name_to_prob = {'network': network_problem(), 'MPC': MPC_problem(), 'ADP': ADP_problem()}
test_combinations = [
    ('network', 'ECOS', 'loops', 0),
    ('MPC', 'OSQP', 'loops', 0),
    ('ADP', 'SCS', 'loops', 0)
]


@pytest.mark.skipif(not cpg_installed, reason='CVXPYgen is not installed')
@pytest.mark.parametrize('name, solver, style, seed', test_combinations)
def test(name, solver, style, seed):
    prob = name_to_prob[name]

    if seed == 0:
        cpg.generate_code(
            prob, code_dir=f'test_{name}_{solver}_{style}', solver=solver,
            unroll=(style == 'unroll'), prefix=f'{name}_{solver}_{style}'
        )
        assert len(glob.glob(os.path.join(f'test_{name}_{solver}_{style}', 'cpg_module.*'))) > 0

    with open(f'test_{name}_{solver}_{style}/problem.pickle', 'rb') as f:
        prob = pickle.load(f)

    prob = assign_data(prob, name, seed)

    val_py, prim_py, dual_py, val_cg, prim_cg, dual_cg, prim_py_norm, dual_py_norm = \
        check(prob, solver, name, get_primal_vec)

    if not np.isinf(val_py):
        assert abs((val_cg - val_py) / val_py) < 0.1

    if prim_py_norm > 1e-6:
        assert np.linalg.norm(prim_cg - prim_py, 2) / prim_py_norm < 0.1
    else:
        assert np.linalg.norm(prim_cg, 2) < 1e-3

    if dual_py_norm > 1e-6:
        assert np.linalg.norm(dual_cg - dual_py, 2) / dual_py_norm < 0.1
    else:
        assert np.linalg.norm(dual_cg, 2) < 1e-3
