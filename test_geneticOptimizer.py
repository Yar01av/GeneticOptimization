from unittest import TestCase
from main_script import GeneticOptimizer, KerasPackageWrapper
import numpy as np

class TestGeneticOptimizer(TestCase):
    # TODO make test data and test model(s)
    testing_network = KerasPackageWrapper.make_test_model([1])

    # Templates
    def _check_one_hot(self, input_data, n_categories, exp_result):
        incoding = KerasPackageWrapper.make_one_hot(input_data, n_categories)

        comparison = np.equal(incoding, exp_result)

        self.assertEquals(np.all(comparison), True)

    """Test for make_one_hot"""
    def test_one_hot1(self):
        self._check_one_hot(np.array([np.array([5])]), 6, np.array([[0, 0, 0, 0, 0, 1]]))

    def test_one_hot2(self):
        self._check_one_hot(np.array([[0]]), 1, np.array([[1]]))

    def test_one_hot3(self):
        self._check_one_hot(np.array([[1], [0]]), 2, np.array([[0, 1], [1, 0]]))

    """Test get_accuracy"""
    def test_get_accuracy(self):
        model = KerasPackageWrapper.make_flat_sequential_model()

        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.2))

        # TODO implement the rest of the test
        print()
