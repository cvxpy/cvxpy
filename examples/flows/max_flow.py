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

# An object oriented max-flow problem.
class Edge(object):
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.in_flow = Variable()
        self.out_flow = Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(self.in_flow)
        out_node.edge_flows.append(self.out_flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.in_flow + self.out_flow == 0,
                abs(self.in_flow) <= self.capacity]

class Node(object):
    """ A node with a target flow accumulation. """
    def __init__(self, accumulation=0):
        self.accumulation = accumulation
        self.edge_flows = []
    
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

if __name__ == "__main__":
    # Read a graph from a file.
    f = open(g.FILE, 'r')
    data = pickle.load(f)
    f.close()

    # Construct nodes.
    node_count = data[g.NODE_COUNT_KEY]
    nodes = [Node() for i in range(node_count)]
    # Add source.
    nodes[0].accumulation = Variable()
    # Add sink.
    nodes[-1].accumulation = Variable()

    # Construct edges.
    edges = []
    for n1,n2,capacity in data[g.EDGES_KEY]:
        edges.append(Edge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])

    # Construct the problem.
    constraints = []
    map(constraints.extend, (o.constraints() for o in nodes + edges))
    p = Problem(Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print result