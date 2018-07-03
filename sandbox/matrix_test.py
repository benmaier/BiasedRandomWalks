import numpy as np
import matplotlib.pyplot as pl
import networkx as nx

from BiasedRandomWalks import BiasedRandomWalk

# Create a test Graph with weighted edges where weights are distances
G = nx.Graph()
N = 4
G.add_nodes_from(range(N))
G.add_edge(0,1,weight=1)
G.add_edge(0,2,weight=1)
G.add_edge(1,2,weight=1)
G.add_edge(1,3,weight=0.1)
G.add_edge(2,3,weight=0.2)

# define sink nodes
sink_nodes = [3,]

# define bias
gamma = 1

# initial distribution on transient
p0 = np.array([1,0,0])

# initial distribution on all
p0_all = np.array([1,0,0,0])

# initial base class (choose 'exponential' or 'scale free')
RW = BiasedRandomWalk(G, gamma, sink_nodes, bias_kind = 'exponential')

print("adjacency_matrix =", RW.adjacency_matrix)
print("weight_matrix =", RW.weight_matrix)
print("transition_matrix =", RW.full_transition_matrix)
print("distance_matrix =", RW.distance_matrix)
