from unittest import TestCase
from main_script import GeneticOptimizer, KerasPackageWrapper
import numpy as np
import keras
from testing_data import get_clean_mnist
from operator import xor


class TestGeneticOptimizer(TestCase):
    # Clean mnist data. Ready to use by the networks
    train_data_x, train_data_y, test_data_x, test_data_y = get_clean_mnist()

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

    def test_inherit_to_child1(self):
        # TODO keras assumed!
        delta = 0.0

        model = KerasPackageWrapper.make_flat_sequential_model()
        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.5))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        child = GeneticOptimizer.inherit_to_child([model], dict(layer_dropout={1}), delta)
        self.assertEqual(model.layers[1].rate, child.layers[1].rate)
        self.assertEqual(model.layers[0].units, child.layers[0].units)

    def test_inherit_to_child2(self):
        # TODO keras assumed!
        delta = 0.2

        model = KerasPackageWrapper.make_flat_sequential_model()
        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.5))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        child = GeneticOptimizer.inherit_to_child([model], dict(layer_dropout={1}), delta)

        if abs(model.layers[1].rate - child.layers[1].rate) >= delta:
            self.fail()
        self.assertEqual(model.layers[0].units, child.layers[0].units)

    def test_inherit_to_child3(self):
        # TODO keras assumed!
        delta = 0.2

        model = KerasPackageWrapper.make_flat_sequential_model()
        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.2))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        child = GeneticOptimizer.inherit_to_child([model], dict(layer_dropout={1}), delta)
        self.assertEqual(model.layers[0].units, child.layers[0].units)
        self.assertAlmostEqual(model.layers[1].rate, child.layers[1].rate, delta = 0.2)
        self.assertEqual(model.layers[2].units, child.layers[2].units)
        self.assertAlmostEqual(model.layers[3].rate, child.layers[3].rate, delta = 0.2)

    def test_inherit_to_child4(self):
        # TODO keras assumed!
        delta = 0.1

        model1 = KerasPackageWrapper.make_flat_sequential_model()
        model1.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model1.add(keras.layers.Dropout(0.2))

        model1.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model2 = KerasPackageWrapper.make_flat_sequential_model()
        model2.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model2.add(keras.layers.Dropout(0.7))

        model2.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        child = GeneticOptimizer.inherit_to_child([model1, model2], dict(layer_dropout={1}), delta)
        # Assert that the structure stays the same as that of parents but the rate changes
        self.assertEqual(child.layers[0].units, model1.layers[0].units)
        self.assertEqual(child.layers[0].units, model2.layers[0].units)

        if abs(child.layers[1].rate - model1.layers[1].rate) >= delta and \
                abs(child.layers[1].rate - model2.layers[1].rate >= delta):
            self.fail()

    # TODO make test_inherit_to_child4 that would not be passed by multiplexer
