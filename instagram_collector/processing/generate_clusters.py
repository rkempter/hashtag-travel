"""
This module handles the processing of clusters
"""
import json

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

    def _compute_radius(center, bounding_box):
        """
        Computes the maximal radius of a cluster around the center of mass, which contains
        all instagrams of the cluster.

        :param center: The center of mass of a cluster
        :param bounding_box: The bounding box around points of a cluster
        :return: radius in meters (float)
        """
        points = [(center.x, bounding_box[0][1]), (center.x, bounding_box[3][1]),
            (bounding_box[0][0], center.y), (bounding_box[3][0], center.y)]

        return max(map(lambda x: great_circle_distance(x[0], x[1]), points))

    cluster = []

    for id, name, center, boundary in result:
        center = loads(center)
        radius = _compute_radius(center, boundary)
        cluster.append((id, name, dumps(center), radius))

    cursor.executemany(insert_query, cluster)
    conn.commit()


def generate_cluster_json(conn):
    """
    Generates a json of the clusters with information about the contained instagrams

    :param conn: Connection to a database
    :return: JSON of all clusters and their corresponding instagrams
    """
    media_query = """
        SELECT
            m.cluster_id,
            c.name,
            ST_AsText(c.center),
            c.radius,
            m.id,
            m.image_url,
            m.location_lat,
            m.location_lng
        FROM
            media_events AS m, cluster AS c
        WHERE
            m.cluster_id = c.id AND
            m.cluster_id IS NOT NULL;"""

    cursor = conn.cursor()

    cursor.execute(media_query)
    media_result = cursor.fetchall()

    cluster = {}

    for cluster_id, cluster_name, center, radius, id, image_url, lat, lng \
            in media_result:

        if cluster_id not in cluster:
            center = loads(center)
            cluster[cluster_id] = {
                "center": {"lat": float(center.y), "lng": float(center.x)},
                "name": cluster_name,
                "radius": radius,
                "media": []
            }

        cluster[cluster_id]['media'].append({
            "id": id,
            "url": image_url,
            "coordinates": {"lat": float(lat), "lng": float(lng)},
        })

    return json.dumps(cluster)