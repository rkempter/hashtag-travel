"""
This module handles the processing of clusters
"""
import logging

import foursquare as fq
import numpy as np
import pandas as pd
from pymongo import MongoClient
from shapely.wkt import dumps, loads
from shapely.geometry import Point, LineString

from instagram_collector.config import (FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET)
from instagram_collector.analytics.collector import connect_postgres_db
from instagram_collector.processing.venue import retrieve_foursquare_data


def great_circle_distance(pnt1, pnt2, radius=6371000):
    """ Similar to great_circle_distance(), but working on list of pnt2 and returning minimum. """
    dLat = np.radians(pnt2[0]) - np.radians(pnt1[0])   # slice latitude from list of (lat, lon) points
    dLon = np.radians(pnt2[1]) - np.radians(pnt1[1])
    a = np.square(np.sin(dLat / 2.0)) + np.cos(np.radians(pnt1[0])) * np.cos(np.radians(pnt2[0])) * np.square(np.sin(dLon / 2.0))
    return np.min(2 * np.arcsin(np.sqrt(a))) * radius

def generate_cluster():
    """
    Clusters all instagrams
    :return:
    """

def update_cluster(conn):
    """
    Update cluster database
    :param conn:
    :return:
    """
    logging.info("Generate location clusters based on database")
    remove_query = """DELETE FROM cluster;"""

    select_query = """
        SELECT
            media.cluster_id, location.location_name, ST_AsText(ST_centroid(ST_Union(geom))),
            ST_AsText(ST_Envelope(ST_Union(geom))), COUNT(DISTINCT(media.user_id)) AS user_count,
            COUNT(DISTINCT(media.id)) AS instagram_count
        FROM
            media_events AS media, (
            SELECT
                pos_table.cluster_id, pos_table.location_name
            FROM (
                SELECT
                    freq_table.cluster_id,
                    freq_table.location_name,
                    ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY freq) AS pos
                FROM (
                    SELECT
                        cluster_id,
                        location_name,
                        COUNT(location_name) AS freq
                    FROM
                        media_events
                    WHERE
                        location_name != '' AND
                        cluster_id IS NOT NULL
                    GROUP BY
                        cluster_id, location_name
                ) AS freq_table
            ) AS pos_table
            WHERE
                pos_table.pos = 1) AS location
        WHERE
            media.cluster_id = location.cluster_id AND
            media.cluster_id IS NOT NULL
        GROUP BY
            media.cluster_id, location.location_name;"""

    insert_query = """
        INSERT INTO
            cluster (id, name, center, radius, user_count, instagram_count)
        VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s);"""

    cursor = conn.cursor()
    cursor.execute(remove_query)
    conn.commit()
    cursor.execute(select_query)
    result = cursor.fetchall()

    def _compute_radius(center, bounding_box):
        """
        Computes the maximal radius of a cluster around the center of mass, which contains
        all instagrams of the cluster.

        :param center: The center of mass of a cluster
        :param bounding_box: The bounding box around points of a cluster
        :return: radius in meters (float)
        """
        shape = loads(bounding_box)
        if isinstance(shape, Point):
            return 0.0
        if isinstance(shape, LineString):
            return 0.0
        else:
            points = [(center.x, shape.exterior.bounds[1]),
                      (center.x, shape.exterior.bounds[3]),
                      (shape.exterior.bounds[0], center.y),
                      (shape.exterior.bounds[2], center.y)]
            return max(map(lambda x: great_circle_distance((center.x, center.y), x), points))

    cluster = []

    for id, name, center, boundary, user_count, instagram_count in result:
        center = loads(center)
        radius = _compute_radius(center, boundary)
        cluster.append((id, name, dumps(center), radius, user_count, instagram_count))

    cursor.executemany(insert_query, cluster)
    conn.commit()


def write_cluster_mongodb(conn, cluster_collection, filter_category_list):
    """
    Generates a json of the clusters with information about the contained instagrams

    :param conn: Connection to a database
    :return: JSON of all clusters and their corresponding instagrams
    """
    logging.info("Write locations to mongo database")
    media_query = """
        SELECT
            m.cluster_id AS cluster_id,
            c.name AS cluster_name,
            ST_AsText(c.center) AS center,
            c.radius AS radius,
            c.instagram_count AS instagram_count,
            c.user_count AS user_count,
            m.id AS id,
            m.image_url AS image_url,
            m.location_lat AS lat,
            m.location_lng AS lng
        FROM
            (SELECT
                id,
                cluster_id,
                image_url,
                location_lat,
                location_lng,
                ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY RANDOM()) AS pos
            FROM media_events) AS m, cluster AS c
        WHERE
            m.cluster_id = c.id AND
            m.cluster_id IS NOT NULL AND
            m.pos < 10;"""

    df_result = pd.read_sql(media_query, conn)
    grouped_cluster = df_result.groupby([
        'cluster_id', 'cluster_name', 'center', 'radius', 'user_count', 'instagram_count'
    ])

    clusters = []

    foursquare_api = fq.Foursquare(client_id=FOURSQUARE_CLIENT_ID,
                                client_secret=FOURSQUARE_CLIENT_SECRET)
    remove_list = []
    for name, group in grouped_cluster:
        cluster_id, cluster_name, center, radius, user_count, instagram_count = name
        center = loads(center)
        group_values = group[['id', 'image_url', 'lat', 'lng']].values
        cluster_data = {}

        data = retrieve_foursquare_data(foursquare_api, cluster_name, center.y, center.x)

        if not data or data['category_name'] in filter_category_list:
            remove_list.append(cluster_id)
            continue;
            # Trying only with locations that we were able to correlate with foursquare
            # data = dict(name=cluster_name, coordinates=[center.y, center.x])

        cluster_data.update(data)
        cluster_data.update({
            "_id": cluster_id,
            "instagram_count": instagram_count,
            "user_count": user_count,
            "radius": radius,
            "topics": [],
            "media": [{"id": media[0],
                       "image_url": media[1]} for media in group_values]
        })
        clusters.append(cluster_data)

    cluster_collection.insert(clusters)

    return remove_list


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)

    # call paris database in mongo db
    mongo_db = client.paris_db
    # connect to postgres server
    connection = connect_postgres_db()
    # Remove all documents from the cluster collection
    mongo_db.cluster_collection.remove({})
    # write the clusters to mongodb
    write_cluster_mongodb(connection, mongo_db.cluster_collection)
