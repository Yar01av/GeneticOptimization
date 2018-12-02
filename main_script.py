# First objective: otimize the drop-outs only
from tensorflow import keras
from random import random

class GeneticOptimizer:
    """The main class for the genetic optimizer"""
    def __init__(self, base_model, training_data, test_data, n_categories , max_deviation=0.2, epochs=3, n_parents=2, traits=None, n_iterations=100):
        # TODO add exception about the traits in relation to the attributes of the model (do the layers
        # given have rates etc.)
        # TODO add exception normalization of the training data
        # TODO check that the model is compiled and do it otherwise
        self.max_deviation = max_deviation
        self.n_categories = n_categories
        self.x_train = training_data[0]
        self.y_train = training_data[1]
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.n_parents = n_parents
        self.epochs = epochs
        self.base_model = base_model
        self.traits = traits  # Hyper parameters to optimize
        self.n_iterations = n_iterations
        self.optimized_models = self.breed([base_model], 1, self.traits)  # A sorted list of models (by performance)
                                                          # after the lest evolutionary step

    """Breeds the parents (already sorted models) and returns a list containing the parents and 
    the mutated children as compiled models."""
    def breed(self, models, n_parents, traits):
        # TODO second
        assert n_parents <= len(models)

        INVARIANT_MODELS = models[:n_parents]  # parents - will remain unchanged
        n_children = len(models) - n_parents
        children = []

        for i in range(n_children):
            pure_child = self.inherit_to_child(INVARIANT_MODELS, traits)
            mutated_child = self.mutate(pure_child, traits, max_deviation=self.max_deviation)

            children.append(mutated_child)

        return INVARIANT_MODELS.concat(children)

    """Returns a list of models where for every compiled model from the 'models' there is a tuple of the trained model
    and the accuracy on test data"""
    def train_models(self, models):
        # TODO second
        trained_models = models[:]

        for model in models:
            trained_model, accuracy = self.do_training(model, self.x_train, self.y_train, self.x_test, self.y_test)
            trained_models.append((trained_model, accuracy))

        return trained_models

    """Modifies a list of models. Architecture remains the same as 'base_model' but hyper-parameters are optimal"""
    def optimize(self):
        # TODO third
        # Let the evolution run for n_iterations
        for i in range(self.n_iterations):
            models = self.breed(self.optimized_models, self.n_parents, self.traits)
            trained_models = self.train_models(models)  # A list of (trained model, accuracy) tuples
            self.optimized_models = sorted(trained_models, key=lambda t1, t2: t2)

    """Returns a list of compiled models with randomized hyper-parameters and the same architecture as the 'model'"""
    def generate_population(self, model):
        # TODO first
        return KerasPackageWrapper.make_flat_sequential_model()

    """Returns a child with a random set of traits (ones passed to the constractor) 
    from its parents (list of networks)"""
    def inherit_to_child(self, parents, traits):
        # TODO second
        return KerasPackageWrapper.make_flat_sequential_model()

    """Returns the same model (pre-compiled) but with slightly altered hyper-parameters (and compiled)"""
    @staticmethod
    def mutate(child, traits_to_alter, max_deviation):
        # TODO first
        mutated_child = child

        # Mutate dropouts
        if "layer_dropout" in traits_to_alter.keys():
            for layer in traits_to_alter["layer_dropout"]:
                # TODO continue
                mutated_child.layers[layer].rate = child.layers[layer].rate * max_deviation * random()


        return child

    """Returns tuple of the compiled 'model' trained on 'training_data' which is assumed normalized 
    and accuracy on the test data"""
    def do_training(self, model, x_train, y_train, x_test, y_test):
        # TODO second
        one_hot_y_train_labels = KerasPackageWrapper.make_one_hot(y_train, self.n_categories)
        one_hot_y_test_labels = KerasPackageWrapper.make_one_hot(y_test, self.n_categories)

        # Keras model assumed
        trained_model = model.fit(x_train, one_hot_y_train_labels, batch_size=200, epochs=self.epochs)
        accuracy = KerasPackageWrapper.get_accuracy(trained_model, x_test, one_hot_y_test_labels)

        return trained_model, accuracy


class KerasPackageWrapper:
    """Wraps around the ML package used - tf.Keras"""
    # TODO make such classes for tflearn and theano. Inherit them from a common abstract class.

    """Returns one hot encoding of the categorical matrix data in 2d numpy array"""
    @staticmethod
    def make_one_hot(categorical_data, n_categories):
        incoding = keras.utils.to_categorical(categorical_data, num_classes=n_categories)

        return incoding

    """Returns a sequential model with random architecture"""
    @staticmethod
    def make_flat_sequential_model():
        return keras.Sequential()

    """Returns accuracy for the given compiled model on the test data (given as one-hot encoding)"""
    @staticmethod
    def get_accuracy(comp_model, x_test, y_test):
        result = comp_model.evaluate(x_test, y_test)

        return result[1]

    """Compiles the given model"""
    @staticmethod
    def compile_model(model):
        return model.compile()

    """Testing related"""
