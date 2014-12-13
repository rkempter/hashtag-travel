"""
Module for computing similarity sets
(now, a location can be in multiple sets)
"""
import numpy as np

from instagram_collector.processing.config import TOPIC_NBR
from instagram_collector.processing.metric import compute_best_set
from operator import itemgetter
from scipy.stats import entropy

def get_location_matrix(location_collection, topic_nbr=TOPIC_NBR):
    """

    :param location_collection:
    :return:
    """
    location_nbr = location_collection.count()
    locations = location_collection.find({}, {"distribution":1, "category_name": 1});

    array_location_map = []

    location_matrix = np.zeros((topic_nbr, location_nbr));

    for index, location in enumerate(locations):
        array_location_map.append(
            (
                (location['category_name'] if 'category_name' in location else ''),
                location["_id"]
            )
        )
        location_matrix[:,index] = location["distribution"]

    return array_location_map, location_matrix


def get_sets(set_collection, location_collection, topic_count=TOPIC_NBR):
    """
    Generate the topic sets
    :param location_matrix:
    :param threshold:
    :return:
    """

    location_map, location_matrix = get_location_matrix(location_collection, topic_count)

    insert_sets = []
    for topic_nbr, topic_distribution in enumerate(location_matrix):
        topic_distribution = [
            (val, location_map[key][0], location_map[key][1])
            for key, val in enumerate(topic_distribution)
        ]
        topic_distribution = sorted(topic_distribution, key=itemgetter(1), reverse=True)
        best_set = compute_best_set(topic_distribution)

        insert_sets.append({
            "_id": topic_nbr,
            "locations": best_set
        })

    set_collection.insert(insert_sets)

