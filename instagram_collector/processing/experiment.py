"""
Module for conducting experiments
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from instagram_collector.collector import connect_postgres_db
from instagram_collector.processing.generate_sets import (generate_connectivity,
    get_agglomerative_clustering, compute_accuracy, generate_user_set,
    write_centroid, get_feature_matrix, kmeans_cluster_locations)
from instagram_collector.processing.analysis import get_centroid_stats, plot_metrics
from pymongo import MongoClient
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


def test_kmeans_clustering(nbr_cluster_range):
    """
    Check kmeans++ clustering for different cluster sizes
    :param nbr_cluster_range:
    :param use_connectivity:
    :return:
    """

    conn = connect_postgres_db()
    client = MongoClient()
    mongo_db = client.paris_db
    features, id_mapping = get_feature_matrix(mongo_db.cluster_collection)

    plot_title = "Kmeans++ clustering"

    mean_i = []
    median_i = []
    minimum_i = []
    maximum_i = []
    std_i = []
    accuracy_i = []

    for nbr_cluster in nbr_cluster_range:
        accuracy, mean, median, minium, maximum, std = \
            do_kmeans_clustering(conn, mongo_db, features,id_mapping, nbr_cluster)

        mean_i.append(mean)
        median_i.append(median)
        minimum_i.append(minium)
        maximum_i.append(maximum)
        std_i.append(std)
        accuracy_i.append(accuracy)

    plot_metrics(plot_title, mean_i, median_i, std_i,
                 minimum_i, maximum_i, accuracy_i, nbr_cluster_range)


def do_kmeans_clustering(conn, mongo_db, features, id_mapping, nbr_cluster):
    """
    Generate a kmeans clustering with :nbr_cluster
    :param nbr_cluster:
    :return:
    """
    sets = kmeans_cluster_locations(features, id_mapping, nbr_cluster)

    mongo_db.centroid_collection.remove({})
    write_centroid(mongo_db.centroid_collection, mongo_db.cluster_collection, sets)
    df_users = generate_user_set(conn)
    locations = map(lambda cluster_set: len(cluster_set['locations']), sets)

    # Compute metrics
    accuracy = compute_accuracy(mongo_db.cluster_collection, df_users)
    mean, median, minimum, maximum, std = get_centroid_stats(locations)

    return accuracy, mean, median, minimum, maximum, std


def test_agglomerative_clustering(nbr_cluster_range, use_connectivity=False):
    """
    Do agglomerative clustering for different numbers of clusters (:nbr_cluster_range). Usage
    of the connectivity matrix is possible (:use_connectivity)
    :param nbr_cluster_range:
    :param use_connectivity:
    :return:
    """
    conn = connect_postgres_db()
    client = MongoClient()
    mongo_db = client.paris_db
    features, id_mapping = get_feature_matrix(mongo_db.cluster_collection)

    connectivity = None
    plot_title = "Agglomerative clustering"
    if use_connectivity:
        connectivity = generate_connectivity(conn, id_mapping)
        plot_title = "Agglomerative clustering with connectivity"

    mean_i = []
    median_i = []
    minimum_i = []
    maximum_i = []
    std_i = []
    accuracy_i = []

    for nbr_cluster in nbr_cluster_range:
        accuracy, mean, median, minium, maximum, std = \
            do_agglomerative_clustering(conn, mongo_db, features,id_mapping, nbr_cluster,
                                        connectivity)
        mean_i.append(mean)
        median_i.append(median)
        minimum_i.append(minium)
        maximum_i.append(maximum)
        std_i.append(std)
        accuracy_i.append(accuracy)

    plot_metrics(plot_title, mean_i, median_i, std_i,
                 minimum_i, maximum_i, accuracy_i, nbr_cluster_range)


def do_agglomerative_clustering(conn, mongo_db, features, id_mapping,
                                  nbr_cluster, connectivity=None):
    """
    Generate a agglomerative clustering
    :param nbr_cluster:
    :param connectivity:
    :return:
    """
    sets = get_agglomerative_clustering(nbr_cluster, features.todense(), id_mapping, connectivity)

    mongo_db.centroid_collection.remove({})
    write_centroid(mongo_db.centroid_collection, mongo_db.cluster_collection, sets)
    df_users = generate_user_set(conn)
    locations = map(lambda cluster_set: len(cluster_set['locations']), sets)
    # Compute metrics
    accuracy = compute_accuracy(mongo_db.cluster_collection, df_users)
    mean, median, minimum, maximum, std = get_centroid_stats(locations)

    return accuracy, mean, median, minimum, maximum, std


def btm_topic_modelling(storage_path):
    from .topics import (clean_tags, generate_btm_topics, write_btm_cluster_vector,
                         write_mongo_btm_topics)
    start_query = """
        SELECT cluster_id, tags
        FROM media_events
        WHERE tags != '';"""
    connection = connect_postgres_db()
    client = MongoClient()
    mongo_db = client.paris_db
    topic_nbr = 80
    training_documents = clean_tags(connection, start_query, btm=True, stop_words=['paris', 'love', 'france'])
    doc2cluster_map = generate_btm_topics(training_documents, storage_path,
                                          mongo_db.topic_collection, mongo_db.cluster_collection,
                                          1, 0.01, 300, 101, topic_nbr)

#    write_mongo_btm_topics(mongo_db.topic_collection, storage_path, topic_number=topic_nbr)
    write_btm_cluster_vector(mongo_db.cluster_collection, storage_path, doc2cluster_map, topic_nbr=topic_nbr)
