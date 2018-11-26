from unittest import TestCase
from main_script import GeneticOptimizer
import numpy as np

class TestGeneticOptimizer(TestCase):
    # def __equal_lists:

    # Templates
    def _check_one_hot(self, input_data, n_categories, exp_result):
        incoding = GeneticOptimizer.KerasPackageWrapper.make_one_hot(input_data, n_categories)

        comparison = np.equal(incoding, exp_result)

        self.assertEquals(np.all(comparison), True)

    """Test for make_one_hot"""
    def test_one_hot1(self):
        self._check_one_hot(np.array([np.array([5])]), 6, np.array([[0, 0, 0, 0, 0, 1]]))

    def test_one_hot2(self):
        self._check_one_hot(np.array([[0]]), 1, np.array([[1]]))

    def test_one_hot3(self):
        self._check_one_hot(np.array([[1], [0]]), 2, np.array([[0, 1], [1, 0]]))
