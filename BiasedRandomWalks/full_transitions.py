import numpy as np
import networkx as nx


def get_full_weight_matrix_and_minimal_distances(G,
                                                 sink_nodes,
                                                 use_inverse_distance_as_adjacency = False,
                                                 return_distance_matrix = False,
                                                ):
    """
    Get the inverse adjacency matrix and for each
    transient node the minimal distance to any of the sinks.

    Parameters
    ==========
    G : networkx.Graph
        Undirected Graph. Edges might have the 'weight' attribute to
        account for distance between nodes
    sink_nodes : list of int
        Contains the indices of sink nodes. It's probably good
        if this is sorted to avoid confusion about later mappings
        but it doesn't have to.
    use_inverse_distance_as_adjacency : bool, default : False
        If this is `True`, instead of the adjacency a_ij, the 
        inverse adjacency a_ij^{-1} is returned.
    return_distance_matrix : bool, default : False
        Return the matrix containing the total distance between 
        nodes all nodes i and j

    Returns
    =======
    A : numpy.array
        An (n x n)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the distance
        between i and j. Contains 0 if i and j are not connected.
    W : numpy.array
        An (n x n)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is 1 if i and j
        are connected and zero otherwise.
    min_distances : numpy.array
        A vector with n entries containing the distance of each
        node i to the _nearest_ sink (might be 0 if i is a sink).
    D : numpy.array
        An (n x n)-matrix containing this total distances d_ij between any=
        two nodes i and j.
    """

    nodes = set(list(G.nodes()))
    N = G.number_of_nodes()
    transient_nodes = list(nodes - set(sink_nodes))
    d = dict(nx.all_pairs_dijkstra_path_length(G))
    D = np.zeros((N,N))

    for i in range(N-1):
        for j in range(i+1,N):
            D[i,j] = d[i][j]
            D[j,i] = d[j][i]

    A = nx.adjacency_matrix(G).toarray()
    A = A.astype(float)
    W = A.copy()

    if use_inverse_distance_as_adjacency:
        W[A>0] = 1/A[A>0]
    else:
        W[A>0] = 1

    min_distances = D[:,sink_nodes].min(axis=1)

    if return_distance_matrix:
        return A, W, min_distances, D
    else:
        return A, W, min_distances

def walkers_on_nodes(T, walker_distribution_on_nodes, tmax, return_walker_distribution = False):
    """
    Parameters
    ==========
    T : numpy.array
        An (n x n)-matrix where n is the number of nodes. Each entry [i,j] is the probability for 
        for a random walker positioned at i to be at node j after the
        next time step.
    walker_distribution_on_transient_nodes : numpy.array
        An n-entry array containing the number (or probability) of walkers
        on nodes at time t0.
    tmax : int
        Integrate the distribution up to this time.
        
    Returns
    =======
    t : numpy.array
        A vector of length k containing the corresponding times for the arrived
        walker distribution.
    rho : numpy.array
        A (k x s)-matrix where k is the number of sampled times and s is
        the number of sink-nodes.
    """
    
    # inital walker distribution
    t = np.arange(tmax)
    p = walker_distribution_on_nodes.copy()
    rhos = [p]
    
    for it in t[1:]:
        p = T.dot(p)
        rhos.append(p)
        
    return t, np.array(rhos)


def get_full_biased_transition_matrix(W, gamma, min_distances, sink_nodes, bias_kind='exponential'):
    """
    Parameters
    ==========
    W : numpy.array
        An (n x n)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the inverse distance
        between i and j. Contains 0 if i and j are not connected.
    gamma : float
        The strength of the bias towards nearest sink nodes with 0 <= gamma
    min_distances : numpy.array
        A vector with n entries containing the distance of each
        node i to the _nearest_ sink (might be 0 if i is a sink).
    sink_nodes : list of int
        Contains the indices of sink nodes. It's probably good
        if this is sorted to avoid confusion about later mappings
        but it doesn't have to.
    bias_kind : str, default : 'exponential'
        `exponential` : The bias will be of form $\\gamma^{d}$ with d being the
                        minimum distance to the sink node
        `scalefree` : The bias will be of form $d^\\gamma$ with d being the
                      minimum distance to the sink node
        
    Returns
    =======
    T : numpy.array
        An (n x n)-matrix where n is the number of nodes. Each entry [i,j] is the probability for 
        for a random walker positioned at i to be at node j after the
        next time step.
    """
    
    nodes = set(range(W.shape[0]))
    transient_nodes = list(nodes - set(sink_nodes))  
    
    if bias_kind == 'exponential':
        assert(0 < gamma and gamma <= 1) 
        alpha = gamma**min_distances
    elif bias_kind == 'scalefree':
        assert(0 >= gamma) 
        d = min_distances.copy()
        d[min_distances == 0.0] = 1
        alpha = d**gamma
    else:
        raise ValueError("Unknown bias_kind '" + str(bias_kind) + "', use 'exponential' or 'scalefree'")
    
    T = W.copy()
    
    #introduce bias
    T *= alpha[:,None]
    k = T.sum(axis=0)

    T /= k[None,:]
    
    for t in transient_nodes:
        for s in sink_nodes:
            T[t, s] = 0
    for s1 in sink_nodes:
        for s2 in sink_nodes:
            if s1 == s2:
                val = 1
            else:
                val = 0
            T[s1,s2] = val
    
    return T
