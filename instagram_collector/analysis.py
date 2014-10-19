"""
Analysis.py contains methods that allow the computation of statistics in the instagram and
foursquare data set.
"""

import pandas as pd
import numpy as np

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
        if len(group) < 2:
            return 0

        group.sort(ascending=True)
        frequency = 0
        for index in range(1, len(group)):
            frequency += group[index] - group[index-1]

        return frequency

    grouped = df.groupby('user_id')
    return grouped['timestamp'].apply(_frequency)

def compute_number_places(df):
    """
    Compute for each user the number of places he has visited
    """
    pass

def compute_average_visit_count(df):
    """
    Compute the median and average of visits at unique places for all users
    """
    pass

def compute_average_trip_length(df):
    """
    Compute the median and average trip length for all users
    """
    pass

def get_user_count(df):
    """
    Return the number of unique users in the dataset
    """
    return len(np.unique(df['user_id'].values))

def get_place_count(df):
    """
    Return the number of unique places in the dataset
    """
    pass

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
