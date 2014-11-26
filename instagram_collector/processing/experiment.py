"""
Module for conducting experiments
"""

import logging

from sklearn.cluster import k_means

def estimate_cluster_nbr(features, range_to_test):
    """
    Estimates the optimal number of clusters
    You can find here a very good explanation about how to do it correctly:
    https://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means
    :return: distortions
    """

    # Need to use a test collection
    distortions = []
    for cluster_nbr in range_to_test:
        centroid, label, inertia = k_means(features, cluster_nbr, init="k-means++")

        distortions.append(inertia)

    return distortions
