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
sink_nodes = [2,]

# define bias
gamma = -0.1

# initial distribution on transient
p0 = np.array([1,0,0])

# initial distribution on all
p0_all = np.array([1,0,0,0])

# initial base class (choose 'exponential' or 'scale free')
RW = BiasedRandomWalk(G, gamma, sink_nodes, bias_kind = 'scalefree', use_additional_adjacency_in_bias=True)

fig, ax = pl.subplots(2,2,figsize=(9,7))

# integrate up to this time step
tmax = 10

# ========== prob density on sinks =========
t, rho = RW.get_amount_of_walkers_arriving_at_sink_nodes(p0,tmax)

for i_s, s in enumerate(sink_nodes):
    ax[0,0].plot(t, rho[:,i_s], label='sink node '+ str(s))

ax[0,0].legend()
ax[0,0].set_xlabel('time')
ax[0,0].set_ylabel('amount of walkers arriving')

# ========== cdf on sinks =========
t, rho = RW.get_amount_of_walkers_arrived_at_sink_nodes(p0,tmax)

for i_s, s in enumerate(sink_nodes):
    ax[0,1].plot(t, rho[:,i_s], label='sink node '+ str(s))

ax[0,1].legend()
ax[0,1].set_xlabel('time')
ax[0,1].set_ylabel('total amount of walkers arrived')

# ========== prob density on all =========
t, rho = RW.get_amount_of_walkers_on_nodes(p0_all,tmax)

for s in G.nodes():
    ax[1,0].plot(t, rho[:,s], label='node '+ str(s))

ax[1,0].legend()
ax[1,0].set_xlabel('time')
ax[1,0].set_ylabel('amount of walkers on each node')


pl.show()
