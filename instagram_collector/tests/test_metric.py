from unittest import TestCase

__author__ = 'rkempter'

from instagram_collector.processing.metric import *
from scipy.stats import entropy


class TestMetric(TestCase):

    def setUp(self):
        self.locations = [
            (0.9, 1, 0),  # 0
            (0.8, 1, 1),  # 1
            (0.7, 2, 2),  # 2
            (0.6, 2, 3),  # 3
            (0.5, 1, 4),  # 4
            (0.5, 4, 5),  # 5
            (0.4, 5, 6),  # 6
            (0.3, 1, 7),  # 7
            (0.2, 5, 8),  # 8
            (0.1, 3, 9)   # 9
        ]

    def test_generate_category_set(self):
        current_set = [0, 1, 2, 3, 4]
        result = generate_category_set(current_set, self.locations)
        self.assertDictEqual(dict(result), {1: 3, 2: 2}, "Category lists are not equal")

        current_set = [0, 1, 2, 5, 4]
        result = generate_category_set(current_set, self.locations)
        self.assertDictEqual(dict(result), {1: 3, 2: 1, 4: 1})

    def test_estimate_constants(self):
        topic_values = map(lambda location: location[0], self.locations[:5])
        lambda_topic, lambda_entropy = estimate_constants(topic_values, set_size=5)
        expected_lambda_topic = 1.0 / (0.5 + 1) * 1.0 / np.sum(topic_values)
        expected_lambda_entropy = 0.5 / (0.5 + 1) * 1.0 / entropy(np.array([0.2] * 5))

        self.assertEqual(lambda_topic, expected_lambda_topic)
        self.assertEqual(lambda_entropy, expected_lambda_entropy)

    def test_get_replace_index(self):
        current_set = [0, 1, 2, 3, 4]
        categories = {1: 3, 2: 2}
        expected_result = [4, 3]
        possibilities = get_replace_index(categories, current_set, self.locations, 5)
        self.assertListEqual(expected_result, possibilities)

        current_set = [0, 1, 2, 5, 6]
        categories = { 1: 2, 2: 1, 4: 1, 5: 1}
        expected_result = [1]
        possibilities = get_replace_index(categories, current_set, self.locations, 5)
        self.assertListEqual(expected_result, possibilities)

    def test_get_topic_scores(self):
        current_set = [0, 1, 2, 3, 4]
        result = get_topic_scores(current_set, self.locations)
        self.assertListEqual([0.9, 0.8, 0.7, 0.6, 0.5], result)

    def test_compute_best_set(self):
        expected_result = [0, 1, 2, 5, 6]

        result_set = compute_best_set(self.locations, 5)
        self.assertListEqual(result_set, expected_result)
