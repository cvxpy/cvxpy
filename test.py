import numpy as np
import scipy.io as sio
import cvxpy as cp
antennas, UE, Bs = 2, 3, 2
H_ = np.mat([[0.05221169+0.02236014j, 0.11507819+0.23286528j, 0.04240248+0.05163106j],
[0.0647228 +0.12858326j, 0.27416864+0.36640208j, 0.01989594+0.07813894j],
[0.62388787+0.62330062j, 0.02298837+0.03943087j, 0.03940723+0.01719871j],
[0.63787985+0.2536698j, 0.16819581+0.2366397j, 0.15092605+0.1541783j]])

sigma = [[1.]]
gamma = 1.
Pc =100.
w = cp.Variable((antennas * Bs, UE), complex=True)
Pw = cp.Variable(Bs)

constr = []
for m in range(UE):
    e = np.mat(np.eye(UE, UE))
    e[m, m] = 0
    t = cp.real(H_[:, m].H @ w[:, m])
    x = cp.hstack((H_[:, m].H @ w @ e, sigma))
    constr += [cp.SOC(t / cp.sqrt(gamma), x), cp.imag(H_[:, m].H @ w[:, m]) == 0]

for q in range(Bs):
    constr += [cp.norm(w[q*antennas:(q+1)*antennas, :], 'fro') <= Pw[q], Pw[q] <= cp.sqrt(Pc)]

print(constr)
prob = cp.Problem(cp.Minimize(cp.sum(Pw)), constr)
print(prob)
print(prob.is_dcp())
prob.solve()
print('Optimal value:', prob.value)
