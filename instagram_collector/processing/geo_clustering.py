"""
Module handles the clustering of geo-located points
"""
import logging
import numpy as np
import psycopg2

from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, MultiPolygon

from instagram_collector.config import MIN_LATITUDE, MIN_LONGITUDE, MAX_LATITUDE, MAX_LONGITUDE
from instagram_collector.processing import config

def get_points(conn, shape_string):
    """
    Get all instagram that lie inside a shape
    :param conn - The PostGreSQL connection (with PostGIS)
    :param shape_string - the string describing the polygon
    :return instagrams
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            id, location_lng, location_lat
        FROM
            media_events
        WHERE
            ST_Contains(
                ST_GeomFromText('Polygon((%s))', 4326),
                geom);
    """ % shape_string)
    conn.commit()
    return cur.fetchall()

def compute_grid(start_pt, end_pt, step_x_count=config.SIZE_X, step_y_count=config.SIZE_Y):
    """
    Generates a grid of polygons
    :param start_pt - left top point
    :param end_pt - right bottom point
    :param step_x_count - how many cells on x scale
    :param step_y_count - how many cells on y scale
    """
    x_coordinates = np.linspace(start_pt[0],end_pt[0], step_x_count + 1)
    y_coordinates = np.linspace(start_pt[1], end_pt[1], step_y_count + 1)

    polygons = []

    for index_x in range(0, step_x_count):
        column = []
        for index_y in range(0, step_y_count):
            column.append(Polygon(
                [(x_coordinates[index_x],y_coordinates[index_y]),
                 (x_coordinates[index_x],y_coordinates[index_y + 1]),
                 (x_coordinates[index_x + 1], y_coordinates[index_y + 1]),
                 (x_coordinates[index_x + 1], y_coordinates[index_y]),
                 (x_coordinates[index_x], y_coordinates[index_y])])
            )
        polygons.append(column)

    return polygons

def generate_shapestring(grid, index_x, index_y):
    """
    Generate a shapestring that describes a polgyon (the area we want)
    :param grid - the grid with all polygons
    :param index_x - the x coordinate of the cell of interest
    :param index_y - the y coordinate of the cell of interest
    :return string
    """

    from shapely.ops import cascaded_union

    length = len(grid) - 1
    height = len(grid[0]) - 1

    next_x = index_x < length
    next_y = index_y < height

    if index_x >= len(grid) or index_y >= len(grid[0]):
        return []

    indexes = [
        (index_x, index_y),
    ]

    if next_x:
        indexes.append((index_x + 1, index_y))
    if next_y:
        indexes.append((index_x, index_y + 1))
    if next_x and next_y:
        indexes.append((index_x + 1, index_y + 1))

    union = cascaded_union([grid[index[0]][index[1]] for index in indexes])
    bounds =  [(union.bounds[0], union.bounds[1]),
               (union.bounds[0], union.bounds[3]),
               (union.bounds[2], union.bounds[3]),
               (union.bounds[2], union.bounds[1]),
               (union.bounds[0], union.bounds[1])]
    return ",".join(
        map(lambda point: "%s %s" % (point[0], point[1]), bounds)
    )

"""
Find clusters of instagrams
"""
from sklearn.cluster import DBSCAN
from operator import itemgetter

update_query = """UPDATE media_events SET cluster_id = %s WHERE id = %s;"""

def geo_clustering(conn, size_x=config.SIZE_X, size_y=config.SIZE_Y,
                   eps=config.EPS, min_pts=config.MIN_PTS):
    """

    :param conn:
    :return:
    """
    # Projection to m
    m = Basemap(
        ellps = 'WGS84',
        projection='merc',
        llcrnrlon=MIN_LONGITUDE,
        llcrnrlat=MIN_LATITUDE,
        urcrnrlon=MAX_LONGITUDE,
        urcrnrlat=MAX_LATITUDE,
        lat_ts=0,
        resolution='i',
        suppress_ticks=True)

    polygons = compute_grid((2.0,47.8),(2.6,49.0))

    conn.commit()
    dbscan = DBSCAN(eps=config.EPS, min_samples=config.MIN_PTS)

    # Generate clusters
    cluster_nbr = 0

    for index_x in range(0, size_x):
        for index_y in range(0, size_y):
            shape_string = generate_shapestring(polygons, index_x, index_y)
            points = get_points(conn, shape_string)
            if len(points) > min_pts:
                points_corrected = \
                    np.array(map(\
                        lambda point: m(float(point[1]), float(point[2])), points)
                    )
                db = dbscan.fit(points_corrected)
                labels = db.labels_[db.labels_ >= 0] + cluster_nbr
                media_id = np.array(map(itemgetter(0), points))
                media_id_sorted = media_id[db.labels_ >= 0]
                update_tuples = zip(labels, media_id_sorted)

                if labels.any() and np.max(labels) >= cluster_nbr:
                    cur = conn.cursor()
                    cur.executemany(update_query, update_tuples)
                    conn.commit()
                    cluster_nbr = np.max(labels) + 1


def pre_processing(conn):
    """
    Set all cluster ids to NULL
    :param conn:
    :return:
    """

    query = """UPDATE media_events SET cluster_id = NULL;"""
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()


def post_processing_user_limit(conn, min_user_count):
    """
    Post processing of clusters. Clusters need to fullfill some quality criterion
    in order to be accepted as clusters. All clusters that have #(users) < min_user_count
    are removed
    :param conn:
    :param min_user:
    :return:
    """

    query = """
        UPDATE media_events
        SET cluster_id = NULL
        WHERE cluster_id IN (
            SELECT cluster_id
            FROM media_events
            GROUP BY cluster_id
            HAVING COUNT(user_id) < %s);"""

    cursor = conn.cursor()
    try:
        cursor.execute(query, (str(min_user_count),))
        conn.commit()
    except psycopg2.Error as e:
        logging.getLogger(__name__).error(e)

