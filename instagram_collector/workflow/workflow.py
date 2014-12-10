__author__ = 'rkempter'

import logging

from instagram_collector.collector import connect_postgres_db
from instagram_collector.processing.generate_clusters import update_cluster, write_cluster_mongodb
from instagram_collector.processing.generate_sets import (get_feature_matrix,
                                                          kmeans_cluster_locations, write_centroid)
from instagram_collector.processing.geo_clustering import (geo_clustering,
                                                           post_processing_user_limit,
                                                           pre_processing)
from instagram_collector.processing.topics import (clean_tags, generate_btm_topics,
                                                   write_mongo_btm_topics, write_btm_cluster_vector,
                                                   write_mongodb_distribution)

from pymongo import MongoClient

def execute_workflow(topic_nbr):
    conn = connect_postgres_db()
    mongo = MongoClient()
    mongo_db = mongo.paris_db
    store_path = "/home/rkempter/research/btm/"
    start_query = """
        SELECT cluster_id, tags
        FROM media_events
        WHERE tags != '';"""

    mongo_db.location_collection.remove({})
    mongo_db.set_collection.remove({})
    mongo_db.topic_collection.remove({})

    # Remove all cluster_ids
    logging.getLogger(__name__).info("Preprocessing")
    pre_processing(conn)


    # Cluster the instagrams using DBSCAN
    logging.getLogger(__name__).info("Start DBSCAN")
    geo_clustering(conn)

    # Do post processing of the clusters based on user limits
    logging.getLogger(__name__).info("Post processing of geo clusters")
    post_processing_user_limit(conn, 20)

    # Update the cluster database
    logging.getLogger(__name__).info("Update of the clusters in the database")
    update_cluster(conn)

    # Write to mongo db
    logging.getLogger(__name__).info("Write locations to mongo db")
    write_cluster_mongodb(conn, mongo_db.location_collection)

    logging.getLogger(__name__).info("Generate training corpus")
    training_documents = clean_tags(conn, start_query, btm=True, stop_words=['paris', 'love', 'france'])

    # Generate the BTM topics
    logging.getLogger(__name__).info("Do BTM")
    doc2cluster_map = generate_btm_topics(training_documents, store_path,
                                          mongo_db.topic_collection, mongo_db.location_collection,
                                          1, 0.01, 400, 101, topic_nbr)


    # Write the topics to mongo db
    logging.getLogger(__name__).info("Write BTM topics to mongo")
    write_mongo_btm_topics(mongo_db.topic_collection, store_path, topic_number=topic_nbr)

    # Write the distribution to the location collection
    logging.getLogger(__name__).info("Get the distribution")
    write_btm_cluster_vector(mongo_db.location_collection, store_path, doc2cluster_map, topic_nbr=topic_nbr)

    # Get features for each location
    logging.getLogger(__name__).info("Generate the feature space for clustering")
    features, id_map = get_feature_matrix(mongo_db.location_collection, topic_nbr)

    # Compute the sets
    logging.getLogger(__name__).info("Cluster using kmeans")
    sets = kmeans_cluster_locations(features, id_map, 60)

    # Write the sets out
    logging.getLogger(__name__).info("Write the sets to mongo db")
    write_centroid(mongo_db.set_collection, mongo_db.location_collection, sets)
