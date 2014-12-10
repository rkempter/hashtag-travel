"""
Module for computing similarity sets
(now, a location can be in multiple sets)
"""
import numpy as np

from instagram_collector.processing.config import TOPIC_NBR


def get_location_matrix(location_collection, topic_nbr=TOPIC_NBR):
    """

    :param location_collection:
    :return:
    """
    location_nbr = location_collection.count()
    locations = location_collection.find({}, {"distribution":1});

    array_location_map = []

    location_matrix = np.zeros((topic_nbr, location_nbr));

    for index, location in enumerate(locations):
        array_location_map.append(location["_id"])
        location_matrix[:,index] = location["distribution"]

    return array_location_map, location_matrix


def get_sets(topic_collection, location_collection, threshold, topic_nbr=TOPIC_NBR):
    """
    Generate the topic sets
    :param location_matrix:
    :param threshold:
    :return:
    """

    location_map, location_matrix = get_location_matrix(location_collection, topic_nbr)

    for topic_nbr, topic_distribution in enumerate(location_matrix):
        topic_set = []
        for location_index, value in enumerate(topic_distribution):
            if value > threshold:
                topic_set.append(location_map[location_index])

        location_collection.update(
            {"_id": { "$in": topic_set}},
            {
                "$push": {
                    "topics": topic_nbr
                }
            }
        )

        topic_collection.update(
            {"_id": topic_nbr},
            {
                "$set": {
                    "location_set": topic_set
                }
            }
        )
