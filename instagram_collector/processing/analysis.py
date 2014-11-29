__author__ = 'rkempter'

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from operator import itemgetter

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


def plot_metrics(plot_title, mean, median, std, minimum, maximum, accuracy, nbr_cluster_range):
    """
    Generates a plot with all the metrics
    :param plot_title: The title of the plot
    :param mean: array
    :param median: array
    :param minimum: array
    :param maximum: array
    :param std: array
    :param accuracy: array
    :param nbr_cluster_range: an array of the number of clusters against which params are plotted
    :return:
    """

    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.7, hspace=0.6)
    fig.suptitle(plot_title, fontsize=14)
    ax1 = fig.add_subplot(321)
    ax1.set_title("Accuracy")
    ax1.plot(nbr_cluster_range, accuracy, "b*-")
    ax2 = fig.add_subplot(322)
    ax2.set_title("Mean")
    ax2.plot(nbr_cluster_range, mean, "b*-")
    ax3 = fig.add_subplot(323)
    ax3.set_title("Median")
    ax3.plot(nbr_cluster_range, median, "b*-")
    ax4 = fig.add_subplot(324)
    ax4.set_title("Std")
    ax4.plot(nbr_cluster_range, std, "b*-")
    ax5 = fig.add_subplot(325)
    ax5.set_title("Minimum")
    ax5.plot(nbr_cluster_range, minimum, "b*-")
    ax6 = fig.add_subplot(326)
    ax6.set_title("Maximum")
    ax6.plot(nbr_cluster_range, maximum, "b*-")

    fig.show()


def get_topic_map(topic_collection):
    """
    Generates a map id -> name for topics
    :param topic_collection:
    :return:
    """
    topics = topic_collection.find({}, {"names": 1})
    topic_map = {}
    for topic in topics:
        topic_map[topic["_id"]] = ", ".join(topic["names"])

    return topic_map

def set_cluster_analysis(centroid_collection, topic_collection, threshold, cluster_method="kmeans"):
    """
    Generates a color map for each cluster, describing the distribution of topics for each
    cluster (centroids)
    :param centroid_collection: mongo_db collection of centroids
    :param threshold: Threshold for topics to get own category, otherwise shown as "Other"
    :return:
    """
    import matplotlib as mpl
    cluster_nbr = centroid_collection.count()
    topic_map = get_topic_map(topic_collection)
    topic_map[101] = "Other"

    # Does currently only work with kmeans++, as they generate the centroids directly
    if cluster_method != "kmeans":
        raise NotImplementedError("Currently only supporting kmeans, as they generate "
                                  "directly centroids. For other clustering algorithms, "
                                  "we first need to generate the centroids based on the "
                                  "locations.")

    centroids = centroid_collection.find({}, {"centroid": 1, "locations": 1})
    topic_nbr = 100 #TODO(@rkempter): Make this dynamic

    # Create discrete colorbar
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,topic_nbr+1,topic_nbr+2)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Other category is equal to 101
    top = 0.9
    fig = plt.figure(figsize=(10, cluster_nbr*2))
    for centroid in centroids:
        topic_distribution = defaultdict(float)
        for topic_nbr, share in enumerate(centroid['centroid']):
            if share > threshold:
                topic_distribution[topic_nbr] += share
            else:
                topic_distribution[101] += share

        order = sorted(topic_distribution.items(), key=itemgetter(1))
        order_topics = map(lambda x: x[0], order)
        order_vals = [0.0]
        order_vals.extend(map(lambda x: x[1], order))
        for index_order in range(1,len(order_vals)):
            order_vals[index_order] += order_vals[index_order-1]

        ax = fig.add_axes([0.05, top, 0.8, 0.2 / cluster_nbr])
        unit = 1.0

        for index, left_pos in enumerate(order_vals[:-1]):
            topic = topic_map[order_topics[index]][:15]
            ax.text(float(left_pos) * unit + threshold / 2, 1.05, "%s." % topic,
                verticalalignment='bottom', horizontalalignment='left',
                rotation=45,
                fontsize=8)
        print order_topics
        print order_vals
        print "======="
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                       norm=norm,
                                       values=order_topics,
                                       boundaries=order_vals,
                                       ticks=order_vals, # optional
                                       spacing='proportional',
                                       orientation='horizontal')
        cb.set_label('Number of locations: %d' % len(centroid['locations']))
        top -= 0.8 / cluster_nbr

    plt.show()
