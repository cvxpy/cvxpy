# Finds the separating hyperplane between two polyhedra.
# Data from Section 8.2.2: Separating polyhedra in 2D in http://cvxr.com/cvx/examples/

from set_variables import Polyhedron
from cvxpy import numpy as np
import cvxpy
import matplotlib.pyplot as plt

n = 2
m = 2*n
A1 = np.matrix("1 1; 1 -1; -1 1; -1 -1")
A2 = np.matrix("1 0; -1 0; 0 1; 0 -1")
b1 = 2*np.ones((m,1))
b2 = np.matrix("5; -3; 4; -2")

poly1 = Polyhedron(A1, b1)
poly2 = Polyhedron(A2, b2)

# Separating hyperplane.
normal,offset = poly1.sep_hyp(poly2)

# Plotting
t = np.linspace(-3,6,100);
p = -normal[0]*t/normal[1] + offset/normal[1]
plt.fill([-2, 0, 2, 0],[0,2,0,-2],'b', [3,5,5,3],[2,2,4,4],'r')
plt.axis([-3, 6, -3, 6])
plt.axes().set_aspect('equal', 'box')
plt.plot(t,p)
plt.title('Separating 2 polyhedra by a hyperplane')
plt.show()