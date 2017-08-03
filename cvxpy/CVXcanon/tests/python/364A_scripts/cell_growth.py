import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import math
import time



# data file for flux balance analysis in systems biology
# From Segre, Zucker et al "From annotated genomes to metabolic flux
# models and kinetic parameter fitting" OMICS 7 (3), 301-316. 
import numpy as np


TIME = 0
# Stoichiometric matrix
#	columns are M1	M2	M3	M4	M5	M6	
# For your interest, the rows correspond to the following equations
#	R1:  extracellular -->  M1
#	R2:  M1 -->  M2
#	R3:  M1 -->  M3
#	R4:  M2 + M5 --> 2 M4
#	R5:  extracellular -->  M5
#	R6:  2 M2 -->  M3 + M6
#	R7:  M3 -->  M4
#	R8:  M6 --> extracellular
#	R9:  M4 --> cell biomass
S = np.matrix("""
	1,0,0,0,0,0;
	-1,1,0,0,0,0;
	-1,0,1,0,0,0;
	0,-1,0,2,-1,0;
	0,0,0,0,1,0;
	0,-2,1,0,0,1;
	0,0,-1,1,0,0;
	0,0,0,0,0,-1;
	0,0,0,-1,0,0
	""").T

m,n = S.shape

vmax = np.matrix("""
	10.10;
	100;
	5.90;
	100;
	3.70;
	100;
	100;
	100;
	100
	""")

v = Variable(n)

ANSWERS = []
constraints = [S*v == 0, v <= vmax, 0<=v]
objective = Maximize(v[-1])
prob = Problem(objective, constraints)

tic = time.time()
val_orig = prob.solve()
toc = time.time()
TIME += toc - tic 

ANSWERS.append(val_orig)
pass #print "Maximal rate: ", val_orig
pass #print "L1: ", constraints[0].dual_value
pass #print "L2: ", constraints[1].dual_value
pass #print "L3: ", constraints[2].dual_value


for i in range(len(vmax)):
	vmax_prime = copy.deepcopy(vmax) 
	vmax_prime[i] = 0
	constraints = [S*v == 0, v <= vmax_prime, 0<=v]
	objective = Maximize(v[-1])
	prob = Problem(objective, constraints)
	
	tic = time.time()
	val = prob.solve()
	toc = time.time()

	TIME += toc - tic
	ANSWERS.append(val)
	if val < .2 * val_orig:
		pass #print "Essential gene:", i	



for i in range(len(vmax)):
	vmax_prime = copy.deepcopy(vmax) 
	for j in range(i+1, len(vmax)):
		vmax_prime[i] = 0
		vmax_prime[j] = 0
		constraints = [S*v == 0, v <= vmax_prime, 0<=v]
		objective = Maximize(v[-1])
		prob = Problem(objective, constraints)
		
		tic = time.time()
		val = prob.solve()
		toc = time.time()

		TIME += toc - tic
		ANSWERS.append(val)
		if val < .2 * val_orig:
			pass #print "Synthetic lethal:", i, j