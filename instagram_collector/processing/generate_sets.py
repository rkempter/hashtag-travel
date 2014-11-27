"""
Module that implements different sets to compute accuracy
"""
import logging
import numpy as np
import pandas as pd

from collections import Counter
from instagram_collector.collector import connect_postgres_db
from operator import itemgetter
from pymongo import MongoClient
from pymongo.bulk import BulkWriteError

from .config import LOCATION_CLUSTER_NBR, TOPIC_NBR

def kmeans_feature_matrix(cluster_collection, topic_nbr=TOPIC_NBR):
    """
    Generate feature matrix for kmeans clustering (preparation step)
    :param cluster_collection:
    :param topic_nbr:
    :return:
    """
    from scipy.sparse import dok_matrix

    nbr_location = cluster_collection.count()
    features = dok_matrix((nbr_location, topic_nbr), dtype=float)

    documents = cluster_collection.find({}, {'distribution': 1})
    doc_mapping = []

    for index, document in enumerate(documents):
        doc_mapping.append(document["_id"])
        if "distribution" not in document:
            continue

        for topic_id, value in document["distribution"].items():
            features[index, int(topic_id)] = float(value)

    return features, doc_mapping

def kmeans_cluster_locations(features, location_map, cluster_nbr=LOCATION_CLUSTER_NBR,
                             max_iter=300, n_init=10, init="k-means++"):
    """

    :param features:
    :param cluster_nbr:
    :return:
    """

    from sklearn.cluster import k_means
    feature_csc = features.tocsr()
    centroids, labels, inertia = k_means(
        feature_csc, cluster_nbr,init=init, max_iter=max_iter,n_init=n_init
    )
    sets = {}
    for index in range(0, len(labels)):
        location_index = location_map[index]
        centroid_index = labels[index]

        if centroid_index not in sets.keys():
            centroid = centroids[centroid_index]
            sets[centroid_index] = {
                "_id": str(centroid_index),
                "centroid": centroid.tolist(),
                "locations": [],
            }

        sets[centroid_index]['locations'].append(location_index)

    return sets.values()

def update_cluster(cluster_collection, locations, centroid_id):
    """
    Write kmeans centroid id to the cluster collection
    :param cluster_collection:
    :param locations:
    :param centroid_id:
    :return:
    """
    bulk = cluster_collection.initialize_unordered_bulk_op()
    bulk.find({"_id": {"$in": locations}}).update({"$set": {"centroid": centroid_id}})
    try:
        bulk.execute()
    except BulkWriteError as bwe:
        logging.getLogger(__name__).error(bwe)


def write_centorid(centroid_collection, cluster_collection, sets):
    """

    :param centroid_collection:
    :param cluster_collection:
    :param sets:
    :return:
    """
    centroid_collection.insert(sets)

    for set in sets:
        update_kmeans_cluster(cluster_collection, set['locations'], set['_id'])


def generate_user_set(conn):
    """
    Retrieve the set of users for accuracy computation. We need users that have more than one (> 1)
    instagrams taken in different location clusters
    :param conn: Connection to postgres database
    :return: pandas dataframe with users
    """

    query = """
        WITH valid_user_id AS (
            SELECT user_id, cluster_id
            FROM
                media_events
            WHERE
                cluster_id IS NOT NULL
            GROUP BY
                user_id, cluster_id
            HAVING COUNT(id) > 1
        )

        SELECT DISTINCT
            m.user_id, m.cluster_id
        FROM
            media_events AS m,
            valid_user_id AS v
        WHERE
            m.user_id = v.user_id AND
            m.cluster_id IS NOT NULL;"""

    return pd.read_sql(query, conn)


def compute_accuracy(cluster_collection, df_users):
    """
    Computation of the accuracy of predicting the set of places a person moves around.
    The max number (> 1) of instagrams that falls into the same cluster is counted for the accuracy
    :param cluster_collection: MongoDB collection of location clusters
    :param df_users: Pandas dataframe of users with corresponding cluster_id
    :return:
    """

    in_set = 0.0
    total = 0.0

    user_grouped = df_users.groupby('user_id')

    for user, group in user_grouped:
        # each cluster is counted only once
        locations = set(group['cluster_id'].values)
        location_nbr = len(locations)

        centroids = map(lambda location: location['centroid'],
                        cluster_collection.find({
                            "_id": {"$in": list(locations)}}, {"_id": 0, "centroid": 1})
        )
        intersection_count = Counter(centroid for centroid in centroids)
        max_arg = max(intersection_count.iteritems(), key=itemgetter(1))[0]

        if intersection_count[max_arg] > 1:
            in_set += intersection_count[max_arg]

        total += location_nbr

    logging.getLogger(__name__).info("Accuracy is %f" % (in_set / total))

    return in_set / total

def generate_connectivity(conn, location_map):
    """
    Generates the connectivity map between different clusters
    :param location_map: a mapping from cluster ids to a range(0, cluster_nbr)
    :return:
    """

    import networkx as nx

    df_cluster = pd.read_sql("""
    SELECT
        m.user_id, m.cluster_id
    FROM
        media_events AS m, cluster AS c
    WHERE
        cluster_id IS NOT NULL AND m.cluster_id = c.id;
    """, conn)

    df_edge = pd.merge(df_cluster, df_cluster, left_on='user_id', right_on='user_id')

    all_edge = np.unique(df_edge[['cluster_id_x', 'cluster_id_y']].values)
    all_edge_tuple = set([(edge[0], edge[1]) for edge in all_edge])

    inverse_map = {val:key for key,val in enumerate(location_map)}
    
    graph = nx.Graph()

    for edge in all_edge_tuple:
        start, end = edge
        if start not in inverse_map or end not in inverse_map:
            continue
        graph.add_edge(inverse_map[start], inverse_map[end])

    return nx.to_scipy_sparse_matrix(graph)

def get_agglomerative_clustering(cluster_nbr, features, location_map, connectivity):
    """

    :return:
    """
    from sklearn.cluster import AgglomerativeClustering

    agglo_clustering = AgglomerativeClustering(cluster_nbr, connectivity=connectivity)
    labels = agglo_clustering.fit_predict(features)
    sets = {}

    for index in range(0, len(labels)):
        location_index = location_map[index]
        centroid_index = labels[index]

        if centroid_index not in sets.keys():
            sets[centroid_index] = {
                "_id": str(centroid_index),
                "locations": [],
            }

        sets[centroid_index]['locations'].append(location_index)

    return sets.values()

if __name__ == '__main__':
    connection = connect_postgres_db()
    client = MongoClient('localhost', 27017)
    mongo_db = client.paris_db

    features, doc_mapping = kmeans_feature_matrix(mongo_db.cluster_collection)
    # @Todo: find correct number of clusters (generate print)
    connectivity = generate_connectivity(connection, doc_mapping)
    sets = get_agglomerative_clustering(40, features, doc_mapping, connectivity)

#    sets = kmeans_cluster_locations(features, doc_mapping, LOCATION_CLUSTER_NBR)
    mongo_db.centroid_collection.remove({})
    write_centorid(mongo_db.centroid_collection, mongo_db.cluster_collection, sets)

    # evaluation of accuracy
    df_users = generate_user_set(connection)
    print compute_accuracy(mongo_db.cluster_collection, df_users)
