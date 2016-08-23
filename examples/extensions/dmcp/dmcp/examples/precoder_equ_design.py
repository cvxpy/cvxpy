__author__ = 'Xinyue'

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from examples.extensions.dmcp.dmcp.dmcp.bcd import bcd

n = 10
m = 15
I = np.eye(n)
C = np.random.randn(m,n) # channel matrix
U, S, V = np.linalg.svd(C)
U = U[:,0:n]
sigma_e = Parameter(1, sign = 'positive') # additive Gaussian noise

A = Variable(n,n) # pre-coder
B = Variable(n,m) # equalizer

#cost = square(norm(B*C*A-I,'fro'))/2+square(sigma_e)*square(norm(B,'fro')) # P(xi=0) = P(xi=1) = 0.5
cost = sum_entries(square(B*C*A-I))/2 + square(sigma_e)*sum_entries(square(B))
obj = Minimize(cost/(m*n))
prob = Problem(obj, [norm(A,'fro')<=10])


SNR = np.power(10,np.linspace(-2,1,20))
sigma_e_value = np.sqrt(float(n)/2/m/SNR)

MSE = []
MSE_lin = []
MSE_noprox = []
BER = []
BER_lin = []
BER_noprox = []
for value in sigma_e_value:
    sigma_e.value = value
    B.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), np.transpose(U))
    A.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), V)
    #prob.solve(method = 'bcd')
    print "======= solution ======="
    print "objective =", cost.value
    # test
    test_t = 1000
    MSE.append(0)
    BER.append(0)
    for t in range(test_t):
        s = np.random.randint(0,2,size=(n,1))
        noise = np.random.randn(m,1)*value
        r = B*(C*A*s+noise)
        MSE[-1] += square(norm(s-r)).value/test_t
        BER[-1] += sum_entries(abs(s-r).value>=0.5).value/float(test_t)
    print "mean squared error =", MSE[-1]
    print "bit error rate =", BER[-1]
    # linear
    B.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), np.transpose(U))
    A.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), V)
    #iter, max_slack = bcd(prob, linearize = True, max_iter = 500)
    print "======= solution ======="
    #print "number of iterations =", iter+1
    print "objective =", cost.value
    # test
    test_t = 1000
    MSE_lin.append(0)
    BER_lin.append(0)
    for t in range(test_t):
        s = np.random.randint(0,2,size=(n,1))
        noise = np.random.randn(m,1)*value
        r = B*(C*A*s+noise)
        MSE_lin[-1] += square(norm(s-r)).value/test_t
        BER_lin[-1] += sum_entries(abs(s-r).value>=0.5).value/float(test_t)
    print "mean squared error =", MSE_lin[-1]
    print "bit error rate =", BER_lin[-1]
    # without proximal
    B.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), np.transpose(U))
    A.value = np.dot(np.dot(np.transpose(V), np.diag(float(1)/np.sqrt(S))), V)
    iter, max_slack = bcd(prob, proximal = False, ep = 1e-4)
    print "======= solution ======="
    print "number of iterations =", iter+1
    print "objective =", cost.value
    # test
    test_t = 1000
    MSE_noprox.append(0)
    BER_noprox.append(0)
    for t in range(test_t):
        s = np.random.randint(0,2,size=(n,1))
        noise = np.random.randn(m,1)*value
        r = B*(C*A*s+noise)
        MSE_noprox[-1] += square(norm(s-r)).value/test_t
        BER_noprox[-1] += sum_entries(abs(s-r).value>=0.5).value/float(test_t)
    print "mean squared error =", MSE_noprox[-1]
    print "bit error rate =", BER_noprox[-1]

plt.figure(figsize=(10,5))
plt.subplot(121)
#plt.semilogy(np.log10(SNR), MSE, 'b-o')
#plt.semilogy(np.log10(SNR), MSE_lin, 'r--^')
plt.semilogy(np.log10(SNR), MSE_noprox, 'b-o')
plt.ylabel('MSE')
plt.xlabel('SNR (dB)')
#plt.legend(["proximal", "prox-linear", "without proximal"])

plt.subplot(122)
#plt.plot(np.log10(SNR), BER, 'b-o')
#plt.semilogy(np.log10(SNR), BER_lin, 'r--^')
plt.semilogy(np.log10(SNR), BER_noprox, 'b-o')
plt.ylabel('Averaged BER')
plt.xlabel('SNR (dB)')
#plt.legend(["proximal", "prox-linear", "without proximal"])
plt.show()