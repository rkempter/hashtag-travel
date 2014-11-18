__author__ = 'rkempter'

import numpy as np

def get_cleaned_positions(graph, positions):
    """
    Prepare position coordinates to draw the graph on a basemap. For this, only nodes that
    are in the graph should be included
    :param graph - the graph that should be printed
    :param positions - a dictionary with positions in format {key: {'lat': val, 'lng': val}}
    """
    return {key:val for key, val in positions.items() if key in graph.nodes()}

def positions_cleaning(positions):
    """
    Generates a positional dictionary from a dataframe that has been transformed to a dictionnary
    :param positions - df[['lat', 'lng']].to_dict() with the corresponding index
    :return {key: {'lat': val, 'lng':val}}
    """
    new_positions = {}
    for key, val in positions['lat'].items():
        new_positions[key] = {}
        new_positions[key]['lat'] = val
    for key, val in positions['lng'].items():
        new_positions[key]['lng'] = val
    return new_positions

def great_circle_distance(pnt1, pnt2, radius=6371000):
    """ Similar to great_circle_distance(), but working on list of pnt2 and returning minimum. """
    dLat = np.radians(pnt2[0]) - np.radians(pnt1[0])   # slice latitude from list of (lat, lon) points
    dLon = np.radians(pnt2[1]) - np.radians(pnt1[1])
    a = np.square(np.sin(dLat / 2.0)) + np.cos(np.radians(pnt1[0])) * np.cos(np.radians(pnt2[0])) * np.square(np.sin(dLon / 2.0))
    return np.min(2 * np.arcsin(np.minimum(np.sqrt(a), len(a)))) * radius