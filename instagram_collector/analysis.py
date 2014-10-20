"""
Analysis.py contains methods that allow the computation of statistics in the instagram and
foursquare data set.
"""
import datetime
import logging

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    filename='instagram_logs.log',level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    pass

def get_adjacent_place_matrix(df):
    """
    Return the adjacent matrix for each lace place mapping.
    """
    pass
