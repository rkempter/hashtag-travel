"""
Module that implements different sets to compute accuracy
"""
import logging
import pandas as pd
import pymongo

from collections import Counter
from instagram_collector.collector import connect_postgres_db
from operator import itemgetter
from pymongo import MongoClient
from pymongo.bulk import BulkWriteError


def kmeans_feature_matrix(cluster_collection, topic_nbr=100):
    """
    Generate feature matrix for kmeans clustering (preparation step)
    :param cluster_collection:
    :param topic_nbr:
    :return:
    """
    from scipy.sparse import dok_matrix

    nbr_location = cluster_collection.count()
    features = dok_matrix((nbr_location,topic_nbr), dtype=float)

    documents = cluster_collection.find({}, {'distribution': 1})
    doc_mapping = map(lambda doc: doc["_id"], documents)

    for index, document in enumerate(documents):
        for topic_id, value in document["distribution"]:
            features[index, topic_id] = value

    return features, doc_mapping

def kmeans_cluster_locations(features, location_map, cluster_nbr, max_iter=300, n_init=10, init="k-means++"):
    """

    :param features:
    :param cluster_nbr:
    :return:
    """

    from sklearn.cluster import k_means

    centroids, labels, inertia = k_means(
        features, cluster_nbr,
        init=init, max_iter=max_iter,
        n_init=n_init
    )

    sets = []
    for index in range(0, len(labels)):
        location_index = location_map[index]
        centroid_index = labels[index]

        if centroid_index not in sets.keys():
            centroid = centroids[centroid_index]
            sets.append({
                "_id": centroid_index,
                "centroid": centroid,
                "locations": [],
            })

        sets[centroid_index]['locations'].append(location_index)

    return sets

def update_kmeans_cluster(cluster_collection, locations, centroid_id):
    """
    Write kmeans centroid id to the cluster collection
    :param cluster_collection:
    :param locations:
    :param centroid_id:
    :return:
    """
    bulk = cluster_collection.initialize_unordered_bulk_op()
    bulk.find({"_id": { $in: locations } })
        .update({ $set: { "centroid": centroid_id } })
    try:
        bulk.execute()
    except BulkWriteError as bwe:
        logging.getLogger(__name__).error(bwe)


def write_kmeans_centorid(centroid_collection, cluster_collection, sets):
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


def compute_kmeans_accuracy(cluster_collection, df_users):
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
                            "_id": { $in: locations} }, {"_id": 0, "centroid": 1})
        )

        intersection_count = Counter(centroid for centroid in centroids)
        max_arg = max(intersection_count.iteritems(), key=itemgetter(1))[0]

        if intersection_count[max_arg] > 1:
            in_set += intersection_count[max_arg]

        total += location_nbr

    logging.getLogger(__name__).info("Accuracy is %f" % (in_set / total))

    return (in_set / total)


if __name__ == '__main__':
    connection = connect_postgres_db()
    client = MongoClient('localhost', 27017)
    mongo_db = client.paris_db

    features, doc_mapping = kmeans_feature_matrix(mongo_db.cluster_collection)
    # @Todo: find correct number of clusters (generate print)
    sets = kmeans_cluster_locations(features, doc_mapping, 100)
    write_kmeans_centorid(mongo_db.centroid_collection, mongo_db.cluster_collection, sets)

    # evaluation of accuracy
    df_users = generate_user_set(connection)
    print compute_kmeans_accuracy(mongo_db.cluster_collection, df_users)
