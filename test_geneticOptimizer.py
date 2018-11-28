from unittest import TestCase
from main_script import GeneticOptimizer, KerasPackageWrapper
import numpy as np
from tensorflow import keras


class TestGeneticOptimizer(TestCase):
    """Fetches and purifies mnist"""

    @staticmethod
    def _get_clean_mnist(self):
        # TODO keras implied!
        data = keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = data.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_test /= 255
        x_train /= 255

        return x_train, y_train, x_test, y_test

    # Clean mnist data. Ready to use by the networks
    train_data_x, train_data_y, test_data_x, test_data_y = _get_clean_mnist()

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

        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model.add(keras.layers.Dropout(0.2))

        # Compile the network
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.evaluate(self.test_data_y, self.test_data_y)

        # TODO implement the rest of the test
        print()
