import numpy as np
import networkx as nx

def get_weight_matrix_and_minimal_distances(G,sink_nodes,use_inverse_distance_as_adjacency=False):
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

    Returns
    =======
    W : numpy.array
        An (n x m)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the inverse distance
        between i and j. Contains 0 if i and j are not connected.
    min_distances : numpy.array
        A vector with n entries containing the distance of each
        node i to the _nearest_ sink (might be 0 if i is a sink).
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
    W = A.astype(float)
    if use_inverse_distance_as_adjacency:
        W[A>0] = 1/A[A>0]

    min_distances = D[:,sink_nodes].min(axis=1)
    mean_min_distance = np.mean(min_distances[transient_nodes])
    min_distances /= mean_min_distance

    W = W[:,transient_nodes]

    return W, min_distances

def get_biased_transition_matrix(W, gamma, min_distances, bias_kind='exponential'):
    """
    Parameters
    ==========
    W : numpy.array
        An (n x m)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the inverse distance
        between i and j. Contains 0 if i and j are not connected.
    gamma : float
        The strength of the bias towards nearest sink nodes with0 < gamma <= 1
    min_distances : numpy.array
        A vector with n entries containing the distance of each
        node i to the _nearest_ sink (might be 0 if i is a sink).
    bias_kind : str, default : 'exponential'
        `exponential` : The bias will be of form $\\gamma^{d}$ with d being the
                        minimum distance to the sink node
        `scalefree` : The bias will be of form $d^\\gamma$ with d being the
                      minimum distance to the sink node

    Returns
    =======
    T : numpy.array
        An (n x m)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the probability for
        for a random walker positioned at i to be at node j after the
        next time step.
    """

    if bias_kind == 'exponential':
        assert(0 < gamma and gamma <= 1) 
        alpha = gamma**min_distances
    elif bias_kind == 'scalefree':
        assert(0 >= gamma) 
        alpha = min_distances**gamma
        if gamma == 0:
            alpha[min_distances == 0.0] = 1
    else:
        raise ValueError("Unknown bias_kind '" + str(bias_kind) + "', use 'exponential' or 'scalefree'")
    
    T = W.copy()

    #introduce bias
    T *= alpha[:,None]
    k = T.sum(axis=0)

    T /= k[None,:]

    return T

def get_transient_matrix_and_absorbing_matrix(T, sink_nodes):
    """
    Parameters
    ==========
    T : numpy.array
        An (n x m)-matrix where m is the number of transient nodes and
        n is the number of nodes. Each entry [i,j] is the probability for
        for a random walker positioned at i to be at node j after the
        next time step.
    sink_nodes : list of int
        Contains the indices of sink nodes. It's probably good
        if this is sorted to avoid confusion about later mappings
        but it doesn't have to.

    Returns
    =======
    Q : numpy.array
        An (m x m)-matrix containing the transition probabilities from
        transient nodes to transient nodes.
    R : numpy.array
        An (s x m)-matrix containing the transition probabilities from
        transient nodes to sink nodes.
    """
    nodes = set(range(T.shape[0]))
    transient_nodes = list(nodes - set(sink_nodes))

    Q = T[transient_nodes,:]
    R = T[sink_nodes,:]

    return Q, R

def walkers_arriving_at_sink_nodes(Q, R, walker_distribution_on_transient_nodes, tmax, return_walker_distribution = False):
    """
    Parameters
    ==========
    Q : numpy.array
        An (m x m)-matrix containing the transition probabilities from
        transient nodes to transient nodes.
    R : numpy.array
        An (s x m)-matrix containing the transition probabilities from
        transient nodes to sink nodes.
    walker_distribution_on_transient_nodes : numpy.array
        An m-entry matrix containing the number (or probability) of walkers
        on transient nodes at time t0.
    tmax : int
        Integrate the probabilities up to this time.
    return_walker_distribution : bool, default : False
        If this is true, an m-entry array will be returned, containing
        the walker distribution on transient nodes at time tmax.

    Returns
    =======
    t : numpy.array
        A vector of length k containing the corresponding times for the arrived
        walker distribution.
    rho : numpy.array
        A (k x s)-matrix where k is the number of sampled times and s is
        the number of sink-nodes.
    p : numpy.array, only if `return_walker_distribution` is True
        An m-entry containing the walker distribution on transient
        nodes at time tmax.
    """

    # get number of sink nodes
    s = R.shape[0]

    # inital walker distribution
    rho0 = np.zeros((s,))

    rhos = [rho0]
    t = np.arange(tmax)
    p = walker_distribution_on_transient_nodes.copy()

    for it in t[1:]:
        rho = R.dot(p)
        rhos.append(rho)

        p = Q.dot(p)

    if return_walker_distribution:
        return t, np.array(rhos), p
    else:
        return t, np.array(rhos)


