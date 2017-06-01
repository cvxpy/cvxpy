"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

# Finds the separating hyperplane between two polyhedra.
# Data from Section 8.2.2: Separating polyhedra in 2D in http://cvxr.com/cvx/examples/

import convex_sets as cs
import numpy as np
import matplotlib.pyplot as plt

n = 2
m = 2*n
A1 = np.matrix("1 1; 1 -1; -1 1; -1 -1")
A2 = np.matrix("1 0; -1 0; 0 1; 0 -1")
b1 = 2*np.ones((m,1))
b2 = np.matrix("5; -3; 4; -2")

poly1 = cs.Polyhedron(A1, b1)
poly2 = cs.Polyhedron(A2, b2)

# Separating hyperplane.
normal,offset = cs.sep_hyp(poly1, poly2)
print normal, offset

# Plotting
t = np.linspace(-3,6,100);
p = -normal[0]*t/normal[1] + offset/normal[1]
p = np.array(p).flatten()
plt.fill([-2, 0, 2, 0],[0,2,0,-2],'b', [3,5,5,3],[2,2,4,4],'r')
plt.axis([-3, 6, -3, 6])
plt.axes().set_aspect('equal', 'box')
plt.plot(t,p)
plt.title('Separating 2 polyhedra by a hyperplane')
plt.show()
