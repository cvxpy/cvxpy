import cvxpy as cp
import numpy as np
from cvxpy.atoms.affine.kron import kron

from cvxpy.atoms.affine.partial_trace import partial_trace

def applychan(chan: np.array, rho: cp.Variable, rep: str, dim: tuple[int, int]):
    tol = 1e-10

    dimA, dimB, dimE = None, None, None
    match rep:
        case 'choi2':
            dimA, dimB = dim
        case 'isom':
            dimA = chan.shape[1]
            dimB = dim[1]
            dimE = int(chan.shape[0]/dimB)
            pass

    match rep:
        case 'choi2':
            arg = chan @ kron(rho.T, np.eye(dimB))
            rho_out = partial_trace(arg, [dimA, dimB], 0)
            return rho_out
        case 'isom':
            rho_out = partial_trace(chan @ rho @ chan.conj().T, [dimB, dimE], 1)
            return rho_out

"""
% Nearest correlation matrix in the quantum relative entropy sense

n = 4;
M = randn(n,n);
M = M*M';
cvx_begin
  variable X(n,n) symmetric
  minimize quantum_rel_entr(M,X)
  diag(X) == ones(n,1)
cvx_end
"""
import cvxpy as cp
import numpy as np
n = 2
np.random.seed(0)
M = np.random.randn(n, n)
M = M @ M.T
X = cp.Variable(shape=(n, n), symmetric=True)
obj = cp.Minimize(cp.quantum_rel_entr(M, X))
# obj = cp.Minimize(cp.quantum_cond_entr(M, X))
cons = [cp.diag(X) == np.ones((n,))]
prob = cp.Problem(obj, cons)
prob.solve(solver='MOSEK')

"""
% Compute lower bound on relative entropy of entanglement (PPT relaxation)

na = 2; nb = 2;
rho = randRho(na*nb); % Generate a random bipartite state rho

cvx_begin sdp
    variable tau(na*nb,na*nb) hermitian;
    minimize (quantum_rel_entr(rho,tau)/log(2));
    tau >= 0; trace(tau) == 1;
    Tx(tau,2,[na nb]) >= 0; % Positive partial transpose constraint
cvx_end
"""

def randH(n: int):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return (A + A.conj().T)/2

def randRho(n: int):
    p = 10 * randH(n)
    p = (p @ p.conj().T)/np.trace(p @ p.conj().T)
    return p

na, nb = (2, 2)
rho = randRho(na * nb)
tau = cp.Variable(shape=(na * nb, na * nb), hermitian=True)
obj = cp.Minimize(cp.quantum_rel_entr(rho, tau, (3,3))/np.log(2))
cons = [tau >> 0, cp.trace(tau) == 1, cp.partial_transpose(tau, [na, nb], 1) >> 0]
prob = cp.Problem(obj, cons)

prob.solve(solver='MOSEK', verbose=True)


"""
% Compute capacity of a cq-channel

% Example 2.16 in
%   "Efficient Approximation of Quantum Channel Capacities" by Sutter et al. (arXiv:1407.8202)
rho1 = [1 0; 0 0]; H1 = quantum_entr(rho1);
rho2 = (1/2) * [1 1; 1 1]; H2 = quantum_entr(rho2);

cvx_begin
    variables p1 p2;
    maximize ((quantum_entr(p1*rho1 + p2*rho2) - p1*H1 - p2*H2)/log(2))
    subject to
        p1 >= 0; p2 >= 0; p1+p2 == 1;
cvx_end
"""

import numpy as np
import cvxpy as cp

rho1 = np.array([[1, 0],
                 [0, 0]])
rho2 = 0.5 * np.ones((2, 2))
H1 = cp.von_neumann_entr(rho1)
H2 = cp.von_neumann_entr(rho2)

p1 = cp.Variable()
p2 = cp.Variable()

obj = cp.Maximize((cp.von_neumann_entr(p1 * rho1 + p2 * rho2) - p1 * H1 - p2 * H2)/np.log(2))
cons = [
    p1 >= 0,
    p2 >= 0,
    p1 + p2 == 1
]

prob = cp.Problem(obj, cons)
prob.solve(solver='MOSEK')


"""
% Quantum capacity of degradable channels

% Example: amplitude damping channel
% na = channel input dimension
% nb = channel output dimension
% ne = channel environment dimension
% nf = degrading map environment dimension
na = 2; nb = 2; ne = 2; nf = 2;

% AD(gamma) = isometry representation of amplitude damping channel
AD = @(gamma) [1 0; 0 sqrt(gamma); 0 sqrt(1-gamma); 0 0];
gamma = 0.2;
U = AD(gamma);

% Unitary representation of degrading map
W = AD ((1-2*gamma)/(1-gamma));

% Ic(rho) = coherent information (see Eq. (13) of paper)
Ic = @(rho) quantum_cond_entr( ...
                W*applychan(U,rho,'isom',[na nb])*W', [ne nf], 2)/log(2);

% Quantum capacity = maximum of Ic (for degradable channels)
cvx_begin sdp
    variable rho(na,na) hermitian
    maximize (Ic(rho));
    rho >= 0; trace(rho) == 1;
cvx_end
"""
na, nb, ne, nf = (2, 2, 2, 2)
AD = lambda gamma: np.array([[1, 0],[0, np.sqrt(gamma)],[0, np.sqrt(1-gamma)],[0, 0]])
gamma = 0.2
U = AD(gamma)

W = AD((1-2*gamma)/(1-gamma))

Ic = lambda rho: cp.quantum_cond_entr(
    W @ applychan(U, rho, 'isom', (na, nb)) @ W.conj().T,
    [ne, nf], 1
)/np.log(2)

rho = cp.Variable(shape=(na, na), hermitian=True)
obj = cp.Maximize(Ic(rho))
cons = [
    rho >> 0,
    cp.trace(rho) == 1
]
prob = cp.Problem(obj, cons)
prob.solve(solver='MOSEK', verbose=True)


"""
% Entanglement-assisted classical capacity of a quantum channel

% Dimensions of input, output, and environment spaces of channel
na = 2; nb = 2; ne = 2;
% AD(gamma) = isometry representation of amplitude damping channel
AD = @(gamma) [1 0; 0 sqrt(gamma); 0 sqrt(1-gamma); 0 0];
U = AD(0.2);
assert(size(U,1) == nb*ne && size(U,2) == na);

cvx_begin sdp
    variable rho(na,na) hermitian;
    maximize ((quantum_cond_entr(U*rho*U',[nb ne]) + ...
                    quantum_entr(TrX(U*rho*U',2,[nb ne])))/log(2));
    subject to
        rho >= 0; trace(rho) == 1;
cvx_end
"""
import cvxpy as cp
import numpy as np

na, nb, ne = (2, 2, 2)
AD = lambda gamma: np.array([[1, 0], [0, np.sqrt(gamma)], [0, np.sqrt(1-gamma)], [0, 0]])
U = AD(0.2)

rho = cp.Variable(shape=(na, na), hermitian=True)
obj = cp.Maximize((cp.quantum_cond_entr(U @ rho @ U.conj().T, [nb, ne]) +
                   cp.von_neumann_entr(cp.partial_trace(U @ rho @ U.conj().T, [nb, ne], 1)))/np.log(2))
cons = [
    rho >> 0,
    cp.trace(rho) == 1
]
prob = cp.Problem(obj, cons)
prob.solve(solver='MOSEK', verbose=True)
