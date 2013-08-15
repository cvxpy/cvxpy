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
from max_flow import Node, Edge
import pickle

# Max-flow with different kinds of edges.
class Directed(Edge):
    """ A directed, capacity limited edge """
    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.in_flow <= 0] + super(Directed, self).constraints()

class LeakyDirected(Edge):
    """ A directed edge that leaks flow. """
    EFFICIENCY = .95
    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.EFFICIENCY*self.in_flow + self.out_flow == 0,
                self.in_flow <= 0,
                abs(self.in_flow) <= self.capacity]

class LeakyUndirected(Edge):
    """ An undirected edge that leaks flow. """
    # Model a leaky undirected edge as two leaky directed
    # edges pointing in opposite directions.
    def __init__(self, capacity):
        self.forward = LeakyDirected(capacity)
        self.backward = LeakyDirected(capacity)
        self.in_flow = self.forward.in_flow + self.backward.out_flow
        self.out_flow = self.forward.out_flow + self.backward.in_flow

    def constraints(self):
        return self.forward.constraints() + self.backward.constraints()

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
        edges.append(Directed(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])

    # Construct the problem.
    constraints = []
    map(constraints.extend, (o.constraints() for o in nodes + edges))
    p = Problem(Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print result