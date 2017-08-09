import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

TIME = 0

h = 1.
g = 0.1
m = 10.
Fmax = 10.
p0 = np.matrix('50 ;50; 100')
v0 = np.matrix('-10; 0; -10')
alpha = 0.5
gamma = 1.
K = 35
ANSWERS = []


ps = []
val = 0
while val != float('inf'):  
	pass #print "K=", K
	v =  Variable( 3, K) 
	p =  Variable( 3, K)  
	f =  Variable( 3, K) 


	# Minimizing fuel
	obj_sum = norm( f[:,0] )
	for i in range(1,K):
		obj_sum += norm( f[:,i] )

	obj = Minimize(obj_sum)
	constraints = [v[:,0] == v0, p[:,0] == p0, p[:,K-1] == 0, v[:,K-1] == 0]


	for i in range(K):
		constraints.append( norm( f[:,i] ) <= Fmax )


	for i in range(1, K):
		constraints.append(  v[:,i] == v[:, i - 1] + (h/m) * f[:, i - 1] - h * g * np.array([0, 0, 1])  )
		constraints.append( p[:,i] == p[:,i - 1] + h * (h/2) * (v[:,i] + v[:,i - 1])  )

	for i in range(K):
		constraints.append( p[2,i] >= alpha * norm( p[1:,i])  )

	
	prob = Problem(obj, constraints)
	ps.append(p)
	tic = time.time()
	val = prob.solve()
	toc = time.time()
	TIME += toc - tic
	ANSWERS.append(val)
	K -= 1

pass #print toc - tic


p = ps[-2]# Last p which was feasible


# use the following code to plot your trajectories
# and the glide cone (don't modify)
# -------------------------------------------------------
# fig = pass #plt.figure()
# ax = fig.gca(projection='3d')

X = np.linspace(-40, 55, num=30)
Y = np.linspace(0, 55, num=30)
X, Y = np.meshgrid(X, Y)
Z = alpha*np.sqrt(X**2+Y**2)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
#Have your solution be stored in p

# ax.plot(xs=p.value[0,:].A1,ys=p.value[1,:].A1,zs=p.value[2,:].A1)
# ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')


pass #plt.title('Minimum time path ')
pass #plt.show()