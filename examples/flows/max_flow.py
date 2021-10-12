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

import pickle

from cvxpy import Maximize, Problem, Variable

from .create_graph import EDGES_KEY, FILE, NODE_COUNT_KEY


# An object oriented max-flow problem.
class Edge:
    """ An undirected, capacity limited edge. """
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.flow = Variable()

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [abs(self.flow) <= self.capacity]

class Node:
    """ A node with accumulation. """
    def __init__(self, accumulation: float = 0.0) -> None:
        self.accumulation = accumulation
        self.edge_flows = []

    # Returns the node's internal constraints.
    def constraints(self):
        return [sum(f for f in self.edge_flows) == self.accumulation]

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
        edges.append(Edge(capacity))
        edges[-1].connect(nodes[n1], nodes[n2])
    # Construct the problem.
    constraints = []
    for o in nodes + edges:
        constraints += o.constraints()
    p = Problem(Maximize(nodes[-1].accumulation), constraints)
    result = p.solve()
    print(result)
