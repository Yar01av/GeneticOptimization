from main_script import KerasPackageWrapper
from tensorflow import keras

"""Fetches and purifies mnist"""
def get_clean_mnist():
    # TODO keras assumed!
    data = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test /= 255
    x_train /= 255

    one_hot_labels_train = KerasPackageWrapper.make_one_hot(y_train, 10)
    one_hot_labels_test = KerasPackageWrapper.make_one_hot(y_test, 10)

    return x_train, one_hot_labels_train, x_test, one_hot_labels_test