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

from cvxpy import *
import create_graph as g
import pickle
import random as r
import cvxopt

# Multi-commodity flow.
# TODO this is entirely wrong
COMMODITIES = 5 # Number of commodities.
r.seed(1)

class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.flow = Variable(COMMODITIES)

    def constraints(self):
        return [abs(self.flow) <= self.capacity]

class Node(object):
    """ A node with no accumulation. """
    def __init__(self):
        self.edges = []
    
    def constraints(self):
        return [sum(e.flow for e in self.edges) == 0]

# Read a graph from a file.
f = open(g.FILE, 'r')
data = pickle.load(f)
f.close()

# Construct nodes and edges.
nodes = [Node() for i in range(data[g.NODE_COUNT_KEY])]
edges = []
for n1,n2,capacity in data[g.EDGES_KEY]:
    edges.append(Edge(capacity))
    nodes[n1].edges.append(edges[-1])
    nodes[n2].edges.append(edges[-1])

# Create random commodity sources and sinks.
sources = []
sinks = []
for i in range(COMMODITIES):
    # A vector to limit the source and sink capacity
    # to a single commodity.
    commodity_vec = cvxopt.matrix(0,(COMMODITIES,1))
    commodity_vec[i] = 1
    source = Edge(commodity_vec*Variable())
    sources.append(source)
    sink = Edge(commodity_vec*Variable())
    sinks.append(sink)
    # Attach source and sink to two random nodes.
    n1,n2 = r.sample(nodes, 2)
    n1.edges.append(source)
    n2.edges.append(sink)
edges += sources + sinks

# Construct the problem.
objective = Maximize(sum(sum(s.flow) for s in sources))
constraints = [s.flow >= 0 for s in sources + sinks]
map(constraints.extend, (o.constraints() for o in nodes + edges))
p = Problem(objective, constraints)
result = p.solve()
print "Objective value = %s" % result
# Show how the flow for each commodity.
for i,s in enumerate(sources):
    print "Flow of commodity %s = %s" % (i, s.flow.value[i])