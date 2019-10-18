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
from .create_graph import FILE, NODE_COUNT_KEY, EDGES_KEY
from .max_flow import Node, Edge
import pickle

# Max-flow with different kinds of edges.
class Directed(Edge):
    """ A directed, capacity limited edge """
    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.flow >= 0, self.flow <= self.capacity]

class LeakyDirected(Directed):
    """ A directed edge that leaks flow. """
    EFFICIENCY = .95
    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.EFFICIENCY*self.flow)

class LeakyUndirected(Edge):
    """ An undirected edge that leaks flow. """
    # Model a leaky undirected edge as two leaky directed
    # edges pointing in opposite directions.
    def __init__(self, capacity):
        self.forward = LeakyDirected(capacity)
        self.backward = LeakyDirected(capacity)

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        self.forward.connect(in_node, out_node)
        self.backward.connect(out_node, in_node)

    def constraints(self):
        return self.forward.constraints() + self.backward.constraints()

if __name__ == "__main__":
    # Read a graph from a file.
    f = open(FILE, 'r')
    data = pickle.load(f)
    f.close()

    # Construct nodes.
    node_count = data[NODE_COUNT_KEY]
    nodes = [Node() for i in range(node_count)]
    # Add source.
    nodes[0].accumulation = Variable()
    # Add sink.
    nodes[-1].accumulation = Variable()

    # Construct edges.
    edges = []
    for n1,n2,capacity in data[EDGES_KEY]:
        edges.append(LeakyUndirected(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])

    # Construct the problem.
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = Problem(Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print(result)
