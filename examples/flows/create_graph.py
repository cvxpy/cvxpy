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

# Construct a random connected graph and stores it as tuples of
# (start node #, end node #, capacity).
from random import choice, sample, random
import pickle

# Constants
FILE = "graph_data"
NODE_COUNT_KEY = "node_count"
EDGES_KEY = "edges"

if __name__ == "__main__":
    N = 20
    E = N*(N-1)/2
    c = 10
    edges = []
    # Start with a line.
    for i in range(1,N):
        edges.append( (i-1,i,c*random()) )
    # Add additional edges.
    for i in range(N,E):
        n1,n2 = sample(range(N), 2)
        edges.append( (n1,n2,c) )
    # Pickle the graph data.
    data = {NODE_COUNT_KEY: N,
            EDGES_KEY: edges}
    f = open(FILE, 'w')
    pickle.dump(data, f)
    f.close()
