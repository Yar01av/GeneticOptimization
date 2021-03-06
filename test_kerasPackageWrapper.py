from unittest import TestCase
import keras
from main_script import KerasPackageWrapper
from testing_data import get_clean_mnist, get_clean_mnist_with_cold_labels


class TestKerasPackageWrapper(TestCase):
    # Clean mnist data. Ready to use by the networks
    train_data_x, train_data_y, test_data_x, test_data_y = get_clean_mnist()

    """Test (non-assertive) get_accuracy"""
    def test_get_accuracy(self):
        model = KerasPackageWrapper.make_flat_sequential_model()

        # TODO keras assumed!
        model.add(keras.layers.Dense(10, activation="relu", input_dim=784))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dropout(0.2))

        # Compile the network
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        result = KerasPackageWrapper.get_accuracy(model, self.test_data_x, self.test_data_y)

        print("Test for get_accuracy. Accuracy is {}".format(result))

    """Test do_training (non-assertive)"""
    def test_do_training(self):
        model = KerasPackageWrapper.make_flat_sequential_model()

        # TODO keras assumed!
        model.add(keras.layers.Dense(500, activation="relu", input_dim=784))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(30, activation="relu"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(30, activation="relu"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(10, activation="softmax"))
        model.add(keras.layers.Dropout(0.2))

        # Compile the network
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(KerasPackageWrapper.do_training(model, *get_clean_mnist_with_cold_labels(), 10, 3))