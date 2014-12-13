"""
Compute the best set using a metric
"""

import numpy as np

from .config import SET_SIZE

from collections import defaultdict
from scipy.stats import entropy


def generate_category_set(current_set, locations):
    """
    Generates a dict with category name as key and number of occurences
    in the current_set as value
    :param current_set:
    :param locations:
    :return:
    """
    categories = defaultdict(int)
    for location in current_set:
        categories[locations[location][1]] += 1

    return categories


def estimate_constants(max_topic_val, set_size, ratio=0.5):
    """
    Computes the regularizer constants for topic and entropy. Very straight forward currently
    :param max_topic_val:
    :param set_size:
    :param ratio:
    :return:
    """
    max_entropy = entropy(np.array([1.0/set_size])*set_size)
    reg_topic = 1.0 / (ratio + 1) * 1.0 / max_topic_val
    reg_entropy = ratio / (ratio + 1) * 1.0 / max_entropy

    return reg_topic, reg_entropy

def compute_metric(lambda_topic, lambda_entropy, category_distribution, topic_scores):
    """
    Computes the value of the metric
    :param lambda_topic:
    :param lambda_entropy:
    :param category_distribution:
    :param topic_scores:
    :return:
    """
    return lambda_topic * np.sum(topic_scores) + \
           lambda_entropy * entropy(category_distribution / 5)

def get_topic_scores(current_set, locations):
    """
    Returns the topic scores of the current set
    :param current_set:
    :param locations:
    :return:
    """
    return map(lambda index: locations[index][0], current_set)

def get_replace_index(categories, current_set, locations, set_size):
    """
    Returns a list of indexes that possibly could be replaced
    :param categories:
    :param current_set:
    :param locations:
    :param set_size:
    :return:
    """
    possibilities = []
    visited_categories = []
    loc_index = set_size - 1
    for location in reversed(current_set):
        category = locations[location][1]
        if categories[category] > 1 and category not in visited_categories:
            possibilities.append(loc_index)
            visited_categories.append(category)
        loc_index = loc_index - 1

    return possibilities

def compute_best_set(locations, set_size=SET_SIZE):
    """
    Select best set with metric
    :param sets: [(value, category_id)]
    :return:
    """

    location_index = 5
    lambda_topic, lambda_entropy = estimate_constants(
        np.sum(
            map(lambda location: location[0], locations[:5])
        )
    )
    current_topic_set = range(0, 5)
    categories = generate_category_set(current_topic_set, locations)

    best_set = current_topic_set
    best_metric_value = compute_metric(
        lambda_topic,
        lambda_entropy,
        np.array(categories.keys(), dtype=float),
        get_topic_scores(current_topic_set, locations)
    )

    while location_index < len(locations) and len(categories.keys()) < set_size:
        possible_replacements = get_replace_index(
            categories, current_topic_set, locations, set_size
        )
        best_result = 0
        best_possibility = 0
        topic_sets = []
        for possibilty, index in enumerate(possible_replacements):
            new_topic_set = current_topic_set[:]
            new_topic_set[index] = locations
            new_category_set = generate_category_set(new_topic_set, locations)
            result = compute_metric(
                lambda_topic,
                lambda_entropy,
                new_category_set,
                get_topic_scores(new_topic_set)
            )
            if result > best_result:
                best_possibility = possibilty
                best_result = result
                topic_sets.append(new_category_set)

        current_topic_set = topic_sets[best_possibility]
        if best_result > best_metric_value:
            best_set = current_topic_set[:]

        location_index += 1

    return best_set
