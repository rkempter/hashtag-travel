__author__ = 'rkempter'

import networkx as nx

def compute_betweenness(graph):
    """
    Computes the betweenness centrality for a graph and puts the value
    as an attribute to each node.
    :param graph - a graph object
    :return graph - a graph with node attributes 'betweenness'
    """
    betweenness = nx.betweenness_centrality(graph, weight='weight')
    nx.set_node_attributes(graph, 'betweenness', betweenness)
    return graph