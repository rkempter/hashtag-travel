__author__ = 'rkempter'

import matplotlib.pyplot as plt
import numpy as np

def print_centroid_stats(locations):
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

    print "Statistics for set of %d clusters" % len(locations)
    print "Mean: %f" % mean
    print "Median: %f" % median
    print "Minimum: %f" % minimum
    print "Maximum: %f" % maximum
    print "Std: %f" % std

    plt.hist(locations, bins=np.linspace(1,42,42))