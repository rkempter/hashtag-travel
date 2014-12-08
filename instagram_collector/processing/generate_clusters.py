"""
This module handles the processing of clusters
"""
import json
import numpy as np
import pandas as pd

from instagram_collector.config import (FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET)
from instagram_collector.collector import connect_postgres_db
from .venue import retrieve_foursquare_data
from pymongo import MongoClient
from foursquare import Foursquare
from shapely.wkt import dumps, loads
from shapely.geometry import Point, Polygon

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

    remove_query = """DELETE FROM cluster;"""

    select_query = """
        SELECT
            media.cluster_id, location.location_name, ST_AsText(ST_centroid(ST_Union(geom))),
            ST_AsText(ST_Envelope(ST_Union(geom)))
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
            cluster (id, name, center, radius)
        VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s);"""

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
        else:
            points = [(center.x, shape.exterior.bounds[1]),
                      (center.x, shape.exterior.bounds[3]),
                      (shape.exterior.bounds[0], center.y),
                      (shape.exterior.bounds[2], center.y)]
            return max(map(lambda x: great_circle_distance((center.x, center.y), x), points))

    cluster = []

    for id, name, center, boundary in result:
        center = loads(center)
        radius = _compute_radius(center, boundary)
        cluster.append((id, name, dumps(center), radius))

    cursor.executemany(insert_query, cluster)
    conn.commit()


def write_cluster_mongodb(conn, cluster_collection):
    """
    Generates a json of the clusters with information about the contained instagrams

    :param conn: Connection to a database
    :return: JSON of all clusters and their corresponding instagrams
    """
    media_query = """
        SELECT
            m.cluster_id AS cluster_id,
            c.name AS cluster_name,
            ST_AsText(c.center) AS center,
            c.radius AS radius,
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
        'cluster_id', 'cluster_name', 'center', 'radius'
    ])

    clusters = []

    foursquare_api = Foursquare(client_id=FOURSQUARE_CLIENT_ID,
                                client_secret=FOURSQUARE_CLIENT_SECRET)

    for name, group in grouped_cluster:
        cluster_id, cluster_name, center, radius = name
        center = loads(center)
        group_values = group[['id', 'image_url', 'lat', 'lng']].values
        cluster_data = {}

        try:
            cluster_data.update(foursquare_api, retrieve_foursquare_data(cluster_name, center.x, center.y))
        except Exception:
            pass

        if not cluster_data:
            cluster_data.update(dict(name=cluster_name, coordinates=[center.x, center.y]))

        cluster_data.update({
            "_id": cluster_id,
            "radius": radius,
            "media": [{"id": media[0],
                       "image_url": media[1]} for media in group_values]
        })
        clusters.append(cluster_data)

    return cluster_collection.insert(clusters)


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
