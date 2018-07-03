import numpy as np
import networkx as nx

from BiasedRandomWalks import get_weight_matrix_and_minimal_distances
from BiasedRandomWalks import get_biased_transition_matrix
from BiasedRandomWalks import get_transient_matrix_and_absorbing_matrix
from BiasedRandomWalks import walkers_arriving_at_sink_nodes
from BiasedRandomWalks import get_full_biased_transition_matrix
from BiasedRandomWalks import get_full_weight_matrix_and_minimal_distances
from BiasedRandomWalks import walkers_on_nodes

class BiasedRandomWalk():

    def __init__(self, G, bias, sink_nodes, use_inverse_distance_as_adjacency=False, bias_kind='exponential', use_additional_adjacency_in_bias=False):

        self.use_inverse_distance_as_adjacency = use_inverse_distance_as_adjacency
        self.G = G
        self.sink_nodes = sink_nodes
        self.bias = bias
        self.bias_kind = bias_kind
        self.use_adj_bias = use_additional_adjacency_in_bias

        self.compute_all_matrices()

    def compute_all_matrices(self):

        self.weight_matrix_to_sinks, self.min_distances_to_sinks, self.adj_distance_to_sinks = \
                get_weight_matrix_and_minimal_distances(
                                                        self.G,
                                                        self.sink_nodes,
                                                        self.use_inverse_distance_as_adjacency,
                                                        True
                                                        )

        if self.use_adj_bias:
            A = self.adj_distance_to_sinks
        else:
            A = None

        self.transition_matrix_to_sinks = get_biased_transition_matrix(self.weight_matrix_to_sinks,
                                                                       self.bias,
                                                                       self.min_distances_to_sinks,
                                                                       self.bias_kind,
                                                                       A
                                                                       )
        self.transient_matrix, self.absorbing_matrix = \
                                          get_transient_matrix_and_absorbing_matrix(
                                                  self.transition_matrix_to_sinks,
                                                  self.sink_nodes
                                                  )

        self.adjacency_matrix, self.weight_matrix, _, self.distance_matrix = get_full_weight_matrix_and_minimal_distances(
                                          self.G,
                                          self.sink_nodes,
                                          self.use_inverse_distance_as_adjacency,
                                          return_distance_matrix = True
                                          )

        if self.use_adj_bias:
            A = self.adjacency_matrix
        else:
            A = None

        self.full_transition_matrix = get_full_biased_transition_matrix(self.weight_matrix, 
                                                                        self.bias,
                                                                        self.min_distances_to_sinks, 
                                                                        self.sink_nodes,
                                                                        self.bias_kind,
                                                                        A,
                                                                        )

    def get_amount_of_walkers_arriving_at_sink_nodes(self,initial_distribution_on_transient_nodes,tmax):

        t, rho = walkers_arriving_at_sink_nodes(self.transient_matrix, 
                                                self.absorbing_matrix,
                                                initial_distribution_on_transient_nodes,
                                                tmax)
        return t, rho

    
    def get_amount_of_walkers_on_nodes(self,initial_distribution_on_nodes,tmax):

        t, rho = walkers_on_nodes(self.full_transition_matrix, initial_distribution_on_nodes, tmax)

        return t, rho

    def get_amount_of_walkers_arrived_at_sink_nodes(self,initial_distribution_on_transient_nodes,tmax):

        t, rho = self.get_amount_of_walkers_arriving_at_sink_nodes(initial_distribution_on_transient_nodes,tmax)

        return t, np.cumsum(rho, axis=0)

    def get_mean_traveled_distance_for_sink_nodes(self,initial_distribution_on_nodes,tmax,norm_distance=True):

        p = initial_distribution_on_nodes / initial_distribution_on_nodes.sum()

        N = self.G.number_of_nodes()
        Pt = [ np.eye(N) ]
        
        T = self.full_transition_matrix
        D = self.adjacency_matrix # this contains the distances between nodes along edges
        PD = T*D

        expected_distance = [ np.zeros((len(self.sink_nodes),)) ]

        for t_all in range(1,tmax):

            this_matrix = np.zeros((N,N))

            for t in range(t_all):
                this_matrix += Pt[t].dot(PD).dot(Pt[-t])

            Pt.append( T.dot(Pt[-1]) )

            d = this_matrix.dot(p)
            #print(d[self.sink_nodes], norm, d[self.sink_nodes] / norm)
            d = d[self.sink_nodes]
            if norm_distance:
                #print(t)
                #print(Pt[-1],'\n',np.linalg.matrix_power(T,t_all))
                norm = Pt[-1].dot(p)
                norm = norm[self.sink_nodes]

                norm[norm == 0] = 1.0

                d /= norm
            expected_distance.append(d+expected_distance[-1])

        return np.array(expected_distance)






if __name__ == "__main__":
    import matplotlib.pyplot as pl
