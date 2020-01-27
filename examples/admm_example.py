"""
Copyright 2013 Steven Diamond

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

from cvxpy import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import scipy as sp
import math

def solveX(data):
	a = data[0:3]
	u = data[3:6]
	z = data[6:9]
	rho = data[9]
	x = Variable(3,1)
	g = square(norm(x - a)) + rho/2*square(norm(x - z + u))
	objective = Minimize(g)
	p = Problem(objective, [])
	result = p.solve()
	return x.value

def main():
	#Solve the following consensus problem using ADMM:
	#Minimize sum(f_i(x)), where f_i(x) = square(norm(x - a_i))

	#Generate a_i's
	np.random.seed(0)
	a = np.random.randn(3, 10)

	#Initialize variables to zero
	x = np.zeros((3,10))
	u = np.zeros((3,10))
	z = np.zeros((3,1))

	rho = 5

	#Run 50 steps of ADMM
	iters = 0
	pool = Pool(processes = 10)
	while(iters < 50):

		#x-update: update each x_i in parallel
		temp = np.concatenate((a,u,np.tile(z, (1,10)),np.tile(rho, (10,1)).transpose()), axis=0)
		xnew = pool.map(solveX, temp.transpose())
		x = np.array(xnew).transpose()[0]

		#z-update
		znew = Variable(3,1)
		h = 0
		for i in range(10):
			h = h + rho/2*square(norm(x[:,i] - znew + u[:,i]))
		objective = Minimize(h)
		p = Problem(objective, [])
		result = p.solve()
		z = np.array(znew.value)

		#u-update
		for i in range(10):
			u[:,i] = u[:,i] + (x[:,i] - z.transpose())[0]

		iters = iters + 1

	pool.close()
	pool.join()

	print(x)

if __name__ == '__main__':
	main()
