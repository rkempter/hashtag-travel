__author__ = 'rkempter'

import matplotlib.pyplot as plt
import numpy as np

def get_centroid_stats(locations):
    """
    Compute the different statistics about the clusters
    :param locations: An array containing the number of locations in each cluster
    :return:
    """
    mean = np.mean(locations)
    median = np.median(locations)
    minimum = np.min(locations)
    maximum = np.max(locations)
    std = np.std(locations)

    return mean, median, minimum, maximum, std
