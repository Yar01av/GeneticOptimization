from unittest import TestCase
from tensorflow import keras
from main_script import KerasPackageWrapper


class TestKerasPackageWrapper(TestCase):
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