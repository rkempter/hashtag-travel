"""
Analysis.py contains methods that allow the computation of statistics in the instagram and
foursquare data set.
"""
import datetime
import logging
from collections import defaultdict
from operator import itemgetter

import numpy as np
import networkx as nx
import matplotlib.collections as col
import matplotlib.pyplot as plt
import pandas as pd
from descartes import PolygonPatch
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, MultiPolygon

from instagram_collector.config import MIN_LATITUDE, MIN_LONGITUDE, MAX_LATITUDE, MAX_LONGITUDE
from instagram_collector.analytics.collector import connect_db



# Configure logging
logging.basicConfig(
    filename='instagram_logs.log',level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Venue(object):
    """
    Venue represents a location.
    :param name - location name
    :param lng - longitude
    :param lat - latitude
    """
    def __init__(self, name, lng, lat):
        self.name = name
        self.lng = lng
        self.lat = lat

    def __str__(self):
        return "%s,%f,%f" % (self.name, self.lng, self.lat)

def load_data(csv_file_path):
    """
    Loads the data from a csv file into a pandas dataframe
    :param csv_file_path
    :return dataframe
    """

    df = pd.read_csv(csv_file_path, sep=";")
    df['timestamp'] = \
        df['created_time'].map(
            lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
                       - datetime.datetime(1970,1,1)).total_seconds())
    return df

def compute_trip_length(df):
    """
    For each user, compute the time between the first and the last picture.
    :param df - dataframe
    :return grouped data frame with user_id and duration
    """
    def _duration(group):
        return group.max() - group.min()

    grouped = df.groupby('user_id')
    return grouped['timestamp'].apply(_duration)

def compute_post_frequency(df):
    """
    Compute the frequency of posting a new picture for each user
    """
    def _frequency(group):
        group.sort(ascending=True)
        frequency = 0
        logging.getLogger(__name__).info("Group length: %d" % len(group))
        values = group.values
        print values
        for index in range(1, len(group)):
            frequency = values[index] - values[(index-1)]

        return float(frequency) / (len(group) - 1)

    grouped = df.groupby('user_id')
    return grouped['timestamp'].apply(_frequency)

def filter_users(df, low_threshold, high_threshold):
    """
    Filtering users that have less posts than low_threshold as well as users that have more posts
    than high_treshold
    :param df - dataframe
    :param low_threshold - only users with count(posts) > low_threshold
    :param high_treshold - only users with count(posts) < high_threshold
    """

    return df.groupby('user_id').filter(
        lambda x: True if x['timestamp'].count() < high_threshold \
                       and x['timestamp'].count() > low_threshold \
                       else False).groupby('user_id')

def get_hashtag_frequency(df):
    """
    Extract all hashtags and count the frequency.
    """
    hashtags_list = defaultdict(int)
    for tags in df[df.tags.notnull()]['tags']:
        for tag in tags.split(','):
            hashtags_list[tag] += 1

    return hashtags_list

def compute_number_places(df):
    """
    Compute for each user the number of places he has visited
    """
    pass

def compute_average_visit_count(df, low_threshold = 0, high_threshold = 1000):
    """
    Compute the median and average of visits at unique places for all users
    :return median, mean
    """
    users_df = filter_users(df, low_threshold, high_threshold)
    place_count = compute_number_places(users_df)
    median = place_count.median()
    mean = place_count.mean()

    return median, mean

def compute_average_trip_length(df, low_threshold = 0, high_threshold = 1000):
    """
    Compute the median and mean trip length for all users
    """
    users_df = filter_users(df, low_threshold, high_threshold)
    trip_length = compute_trip_length(users_df)
    median = trip_length.median()
    mean = trip_length.mean()

    return median, mean

def get_user_count(df):
    """
    Return the number of unique users in the dataset
    """
    return len(np.unique(df['user_id'].values))

def get_place_count(df):
    """
    Return the number of unique places in the dataset
    """
    return len(np.unique(df['venue_id'].values))

def get_adjacent_user_matrix(df):
    """
    Return the adjacent matrix for each each user mapping the user - user relations
    """

def get_sample(query):
    """
    Execute a query and load the result into a dataframe.
    @TODO: Use read_sql
    """
    import pandas.io.sql as psql

    mysql_cn = connect_db()
    df = psql.frame_query(query, con=mysql_cn)
    mysql_cn.close()
    return df

def get_adjacent_place_matrix(df):
    """
    Return the adjacent matrix for each place place mapping.
    """
    user_group = df.groupby('user_id')
    user_venue = user_group['venue_id']

    adj_list = defaultdict(int)

    for group in user_venue:
        df_1 = pd.DataFrame(group)
        df_2 = pd.DataFrame(group)
        df_1[0] = df_2[0] = 0
        merged = pd.merge(df_1, df_2, on=0)
        merged = merged[merged['venue_id_x'] != merged['venue_id_y']]
        for edge in merged[['venue_id_x', 'venue_id_y']].values:
            adj_list[(edge[0], edge[1])] += 1

    return adj_list

def create_graph(adj_list):
    """
    Create a graph g based on an adjacent list
    """
    G = nx.Graph()

    for edge, weight in adj_list:
        G.add_edge(edge[0], edge[1], weight)

    return G

def generate_boxes(nbr_lat, nbr_lng):
    """
    Generate boxes for computation of conditional probabilities for a baseline prediction
    """
    length_lng = MAX_LONGITUDE - MIN_LONGITUDE
    length_lat = MAX_LATITUDE - MIN_LATITUDE

    polygons = []

    latitudes = np.linspace(MIN_LATITUDE, MAX_LATITUDE, nbr_lat)
    longitudes = np.linspace(MIN_LONGITUDE, MAX_LONGITUDE, nbr_lng)

    for lat_index in xrange(0, len(latitudes) - 1):
        bottom = latitudes[lat_index]
        top = latitudes[lat_index + 1]
        for lng_index in xrange(0, len(longitudes) - 1):
            left = longitudes[lng_index]
            right = longitudes[lng_index + 1]
            points = [(left, bottom), (left, top), (right, top), (right, bottom)]
            polygons.append(Polygon(points))

    return polygons

def get_patch(shape, **kwargs):
    """Return a matplotlib PatchCollection from a geometry generated by shapely."""
    # Simple polygon.
    if isinstance(shape, Polygon):
        return col.PatchCollection([PolygonPatch(shape, **kwargs)],
                                   match_original=True)
    # Collection of polygons.
    elif isinstance(shape, MultiPolygon):
        return col.PatchCollection([PolygonPatch(c, **kwargs)
                                    for c in shape],
                                   match_original=True)

def print_nodes(graph, metric_dict, positions, fig_title, cmap):

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

    # convert lat and lng to map projection

    def _order_nodes(metric_node_dict):
        """
        :return nodes, colors, positions
        """
        sorted_vals = sorted(metric_node_dict.items(), key=itemgetter(1), reverse=False)
        values = map(itemgetter(1), sorted_vals)
        nodes = map(itemgetter(0), sorted_vals)
        colors = np.array(values, dtype=float) / np.max(values)
        return nodes, colors

    metric_nodes, metric_colors = _order_nodes(metric_dict)
    positions_corrected = {
        key:m(val['lat'], val['lng'])
        for key, val in positions.items()
        if key in graph.nodes()
    }

    nodes = nx.draw_networkx_nodes(
        graph, positions_corrected,
        node_list=metric_nodes,
        node_color=metric_colors,
        cmap=plt.cm.jet,
        node_size=50)

    plt.title(fig_title)
    plt.colorbar(nodes)
    plt.show()

def print_graph(graph, positions={}, colors={}, fig_title="", cmap_name="hot", **kwargs):
    """
    Print a colored graph on a map
    """
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

    # convert lat and lng to map projection
    positions_corrected = {key:m(val['lng'], val['lng']) for key, val in positions.items()}

    cmap = plt.get_cmap(cmap_name)
    nx.draw_networkx(
        graph, positions_corrected,
        node_size=30,
        cmap=cmap,
        node_color=colors,
        with_labels=False
    )
    plt.title(fig_title)
    plt.show()

def plot_map(shapes, points):
    """
    Plot a map with all the shapes and points as pyplot
    """
    fig = plt.figure()
    m = Basemap(
        ellps = 'WGS84',
        projection = 'merc',
        llcrnrlon=MIN_LONGITUDE,
        llcrnrlat=MIN_LATITUDE,
        urcrnrlon=MAX_LONGITUDE,
        urcrnrlat=MAX_LATITUDE,
        resolution='h')
    m.drawmapboundary(fill_color='white')

    ax = plt.subplot(111);
    for shape in shapes:
        ax.add_collection(get_patch(shape))

    plt.title("Grid");

    plt.show()

def get_box_shape(nbr_lat, nbr_lng, index):
    """
    Returns the shape of the box with index index
    """
    pass

def get_grid_index(nbr_lat, nbr_lng, point_lat, point_lng):
    latitudes = np.linspace(MIN_LATITUDE, MAX_LATITUDE, nbr_lat + 1)
    longitudes = np.linspace(MIN_LONGITUDE, MAX_LONGITUDE, nbr_lng + 1)

    def _find_nearest(array,value):
        return (np.abs(array-value)).argmin()

    return _find_nearest(latitudes, point_lat) * nbr_lng + _find_nearest(longitudes, point_lng)

def generate_feature_space(data, nbr_lat, nbr_lng):
    """
    Generates a feature space depending on the given shapes
    1) Retrieve a random set of users that has more than 2 photos
    2) Transform those users into feature vectors with a label
    """

    def _create_feature(group):
        """
        Depending on the grid, generate the feature vector.
        :param group: [['longitude', 'latitude', 'timestmap']]
        """
        group.sort('timestamp', ascending=True)
        vector = np.zeros(nbr_lat * nbr_lng)
        target = get_grid_index(nbr_lat, nbr_lng, group.values[-1][1], group.values[-1][0])
        for lng, lat, timestamp in group.values[:-1]:
            index = get_grid_index(nbr_lat, nbr_lng, lat, lng)
            vector[index] = 1

        return vector, target

    # Generate a python dataframe and group it by the user
    df = pd.DataFrame(data, header=['user_id', 'timestamp', 'location_lat', 'location_lng'])
    user_group = df[['location_lat', 'location_lng', 'timestamp', 'user_id']].groupby('user_id')

    samples = []
    targets = []

    for name, group in user_group:
        feature_vector, target = _create_feature(
            group[['location_lng', 'location_lat', 'timestamp']]
        )
        samples.append(feature_vector)
        targets.append(target)

    return samples

def train_naive_bayes(samples, targets):
    """
    Train a naive bayes classifier
    """
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(samples, targets)

    return clf

def create_test_train(nbr_lat, nbr_lng, size_train, size_test):
    """
    Generate a train / test dataset
    :param nbr_lat - The number of elements on latitude axis
    :param nbr_lng - The number of elements on longitude axis
    :param size_train - A float describing the percentage of the train set
    :param size_test  - A float describing the percentage used for test
    """

    query = """
        SELECT user_id, timestamp, location_lat, location_lng
        FROM media_events
        WHERE user_id IN (
            SELECT user_id
            FROM media_events
            GROUP BY user_id
            HAVING COUNT(*) > 2 LIMIT 30000);
    """

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(query)

    result = [result for result in cursor.fetchall()]
    features, targets = generate_feature_space(result, nbr_lat, nbr_lng)
    random_vector = np.ranlen(features)


    return

def train_test(nbr_lat, nbr_lng, size_train, size_test):
    """
    1) Generate a train and test dataset
    2) Train a naive bayes classifier
    3) Test and compute different metrics (accuracy, distance to the correct one, ...)
    """
