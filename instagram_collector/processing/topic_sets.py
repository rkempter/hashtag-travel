"""
Module for computing similarity sets
(now, a location can be in multiple sets)
"""
import numpy as np

from instagram_collector.processing.config import TOPIC_NBR
from operator import itemgetter
from scipy.stats import entropy

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


def compute_best_set(locations, set_size=SET_SIZE):
    """
    Select best set with metric
    :param sets: [(value, category_id)]
    :return:
    """


    def _generate_category_set(current_set):
        categories = defaultdict(int)
        for location in current_set:
            categories[sets[location][1]] += 1

        return categories

    def _estimate_constants(max_topic_val, ratio=0.5):
        max_entropy = entropy(np.array([1.0/set_size])*set_size)
        reg_topic = 1.0 / (ratio + 1) * 1.0 / max_topic_val
        reg_entropy = ratio / (ratio + 1) * 1.0 / max_entropy

        return reg_topic, reg_entropy

    def _compute_metric(lambda_topic, lambda_entropy, category_distribution, topic_scores):
        return lambda_topic * np.sum(topic_scores) + \
               lambda_entropy * entropy(category_distribution / 5)

    def _get_topic_scores(current_set):
        return map(lambda index: locations[index][0], current_set)

    def _get_replace_index(categories, current_set):
        possibilities = []
        visited_categories = []
        loc_index = set_size - 1
        for location in reversed(current_set):
            category = locations[location][1]
            if categories[category] > 1 and category not in visited_categories:
                possibilities.append(index)
                visited_categories.append(category)
            loc_index = loc_index - 1

        return possibilities

    location_index = 6
    lambda_topic, lambda_entropy = _estimate_constants(
        np.sum(
            map(lambda location: location[0], locations[:5])
        )
    )
    current_topic_set = range(0, 5)
    categories = _generate_category_set(current_topic_set)

    best_set = current_topic_set
    best_metric_value = _compute_metric(
        lambda_topic,
        lambda_entropy,
        np.array(categories.keys(), dtype=float),
        _get_topic_scores(current_topic_set)
    )

    while location_index < len(locations) and len(categories.keys()) < set_size:
        possible_replacements = _get_replace_index(categories, current_topic_set)
        best_result = 0
        best_possibility = 0
        topic_sets = []
        for possibilty, index in enumerate(possible_replacements):
            new_topic_set = current_topic_set[:]
            new_topic_set[index] = locations
            new_category_set = _generate_category_set(new_topic_set)
            result = _compute_metric(
                lambda_topic,
                lambda_entropy,
                new_category_set,
                _get_topic_scores(new_topic_set)
            )
            if result > best_result:
                best_possibility = possibilty
                best_result = result
                topic_sets.append(new_category_set)

        current_topic_set = topic_sets[best_possibility]
        if best_result > best_metric_value:
            best_set = current_topic_set[:]

    return best_set

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
        topic_distribution = {key: val for key, val in enumerate(topic_distribution)}
        topic_distribution = sorted(topic_distribution.items(), key=itemgetter(1), reverse=True)
        for index, topic_tuple in enumerate(topic_distribution):
            location_index, value = topic_tuple
            if index < 5 or value > threshold:
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
