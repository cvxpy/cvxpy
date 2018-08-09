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
