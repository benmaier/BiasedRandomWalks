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
gamma = 1.0

# initial distribution on transient
p0 = np.array([1,0,0])

# initial distribution on all
p0_all = np.array([1,0,0,0])

# initial base class (choose 'exponential' or 'scale free')
RW = BiasedRandomWalk(G, gamma, sink_nodes, bias_kind = 'exponential')

fig, ax = pl.subplots(1,1,figsize=(4,3))

# integrate up to this time step
tmax = 10

# ========== cdf on sinks =========
t, rho = RW.get_amount_of_walkers_arrived_at_sink_nodes(p0,tmax)
d_traveled = RW.get_mean_traveled_distance_for_sink_nodes(p0_all,tmax)
print(d_traveled)

for i_s, s in enumerate(sink_nodes):
    ax.step(d_traveled[:,i_s], rho[:,i_s], label='sink node '+ str(s),where='post')

ax.legend()
ax.set_xlabel('mean traveled distance')
ax.set_ylabel('total amount of walkers arrived')

fig.tight_layout()


pl.show()
