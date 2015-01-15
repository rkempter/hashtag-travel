from collector import connect_postgres_db

import numpy as np
from scipy.sparse import dok_matrix
from shapely.geometry import Point

# Algorithm for computation of distance matrix
#
# 1) Divide the space into a grid
# 2) Compute the distances between each point in the space and with the grid spaces that are around
# 3) Finally, send elements to fitting
#
# Grid: Upper left is 0, lower right is N
#
# squares to take into account:
# [(i-1,j-1), (i-1,j), (i-1,j+1), (i,j-1), (i,j), (i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1)]
#
# def
#
# Order by ID:
# (k,l) = 3 = (l,k)

# def get_points(i,j):
# 	top_left = (i-1 if i > 0 else i, j-1 if j > 0 else j)
# 	bottom_left = (i-1 if i > 0 else i, j+1 if j < max-1 else j)
# 	top_right = (i+1 if i < 0 - 1 else i, j-1 if j > 0 else j)
# 	bottom_right = (i+1 if i < max-1 else i, j+1 if < max-1 else j)
# 	query = """ """ % coordinates of square
# 	return [(id, lat, lng)]

def generate_polygon(x_index, y_index, grid):
    """
    Return a polygon from which we should load all the points
    """
    from shapely.ops import cascaded_union
    length = len(grid[0])

    indexes = [
        (x_index, y_index),
        (x_index+1 if x_index+1 < length else x_index, y_index),
        (x_index, y_index+1 if y_index + 1 < len(grid) else y_index),
        (x_index+1
            if x_index+1 < length else x_index,
         y_index+1
            if y_index + 1 < len(grid) else y_index),
    ]
    polygon = cascaded_union([grid[index[0]][index[1]] for index in indexes])
    coord_string = ",".join(
        map(lambda point: "%s %s" % (point[0], point[1]), list(polygon.exterior.coords))
    )
    return coord_string


def get_points(x_index, y_index, grid):
    query = """
        SELECT *
        FROM
            media_events
        WHERE
            ST_Contains(
                ST_GeomFromText('Polygon((%s))', 4326),
                geom);
    """

    polygon_string = generate_polygon(x_index, y_index, grid)

    conn = connect_postgres_db()
    cursor = conn.cursor()
    cursor.execute(query % polygon_string)
    result = cursor.fetchall()
    conn.close()

    return result


def great_circle_distance(pnt1, pnt2, radius=6371000):
    """ Similar to great_circle_distance(), but working on list of pnt2 and returning minimum. """
    dLat = np.radians(pnt2.x) - np.radians(pnt1.x)   # slice latitude from list of (lat, lon) points
    dLon = np.radians(pnt2.y) - np.radians(pnt1.y)
    a = np.square(np.sin(dLat / 2.0)) + np.cos(np.radians(pnt1.x)) * np.cos(np.radians(pnt2.x)) * np.square(np.sin(dLon / 2.0))
    return np.min(2 * np.arcsin(np.minimum(np.sqrt(a), len(a)))) * radius

def compute_distances(dist_matrix, points):
    """ compute the distance matrix"""
    pair_hash = []
    for index_1, name_1, lat_1, lng_1 in points:
        for index_2, name_2, lat_2, lng_2 in points:
            point_1 = Point(lat_1, lng_1)
            point_2 = Point(lat_2, lng_2)

            if (point_1, point_2) not in pair_hash or \
                (point_2, point_1) not in pair_hash:
                dist_matrix[index_1, index_2] = dist_matrix[index_2, index_1] = \
                    point_1.distance(point_2)

def compute_dist_matrix(nbr_points, grid_size_x, grid_size_y):
    grid = generate_grid(grid_size_x, grid_size_y)
    dist_matrix = dok_matrix(nbr_points, nbr_points)
    for x_index in range(0, grid_size_x):
        for y_index in range(0, grid_size_y):
            points = get_points(x_index, y_index, grid)
            compute_distances(dist_matrix, points)


