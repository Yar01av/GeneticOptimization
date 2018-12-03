# First objective: otimize the drop-outs only
#
# Dropouts should be passed as the dictionary of traits. For example, traits = dict(layer_dropout={1, 3})
# says that the first and third layers (which should be dropout layers) should be optimized
import keras
from random import random
from copy import deepcopy


class GeneticOptimizer:
    """The main class for the genetic optimizer"""
    def __init__(self, base_model, training_data, test_data, n_categories, max_deviation=0.2, epochs=3, n_parents=2, traits=None, n_iterations=100):
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

        trained_initial_model = self.train_models(base_model)  # A list of models (parent and the children)
        self.optimized_models = self.generate_sorted_population(trained_initial_model, self.n_parents, self.traits)
        # A sorted list of models (by performance) after the lest evolutionary step

    """Breeds the parents (already sorted models) and returns a list containing the parents and 
    the mutated children as compiled models."""
    @staticmethod
    def breed(models, n_parents, traits, max_dropout_change_deviation):
        # TODO second
        assert n_parents <= len(models)

        INVARIANT_MODELS = models[:n_parents]  # parents - will remain unchanged
        n_children = len(models) - n_parents
        children = []

        for i in range(n_children):
            pure_child = GeneticOptimizer.inherit_to_child(INVARIANT_MODELS, traits)
            mutated_child = GeneticOptimizer.mutate(pure_child, traits, max_dropout_change_deviation)

            children.append(mutated_child)

        return INVARIANT_MODELS.concat(children)

    """Returns a list of models where for every compiled model from the 'models' there is a tuple of the trained model
    and the accuracy on test data"""
    def train_models(self, models):
        # TODO second
        trained_models = models[:]

        for model in models:
            trained_model, accuracy = KerasPackageWrapper.do_training(model, self.x_train, self.y_train, self.x_test,
                                                                      self.y_test, self.n_categories, self.epochs)
            trained_models.append((trained_model, accuracy))

        return trained_models

    """Modifies a list of models. Architecture remains the same as 'base_model' but hyper-parameters are optimal"""
    def optimize(self):
        # TODO third
        # Let the evolution run for n_iterations
        for i in range(self.n_iterations):
            self.optimized_models = self.generate_sorted_population(self.optimized_models, self.n_parents, self.traits)

    """Returns a list of compiled models with randomized hyper-parameters and the same architecture as the 'model'"""
    def generate_sorted_population(self, optimized_models, n_parents, traits):
        # TODO second
        # TODO continue...
        models = GeneticOptimizer.breed(optimized_models, n_parents, traits, self.max_deviation)
        trained_models = self.train_models(models)  # A list of (trained model, accuracy) tuples

        return sorted(trained_models, key=lambda t1, t2: t2)

    """Returns a child with a random set of traits (ones passed to the constructor) 
    from its parents (list of networks)"""
    @staticmethod
    def inherit_to_child(parents, traits):
        # TODO first
        return KerasPackageWrapper.make_flat_sequential_model()

    """Returns the same model (pre-compiled) but with slightly altered hyper-parameters (and compiled). The altered 
    values should equal the old ones +/- max_deviation"""
    @staticmethod
    def mutate(child, traits_to_alter, max_deviation):
        # TODO second
        mutated_child = KerasPackageWrapper.deep_copy(child)

        # Mutate dropouts
        if "layer_dropout" in traits_to_alter.keys():
            for layer in traits_to_alter["layer_dropout"]:
                # TODO keras assumed
                mutated_child.save("saved_weights.h5")

                # Compute new rate
                mutated_child.layers[layer].rate = GeneticOptimizer._get_mutated_dropout(child.layers[layer].rate,
                                                                                         max_deviation)

                # Propagate the new changes into the backend (cloning is necessary for that)
                mutated_child = keras.models.clone_model(mutated_child)
                mutated_child.load_weights("saved_weights.h5")

        return mutated_child

    """Compute new dropout rate for the given rate"""
    @staticmethod
    def _get_mutated_dropout(old_dropout_rate, deviation):
        # TODO first
        change_by = (random() * 2 * deviation) - deviation
        new_rate = old_dropout_rate - change_by

        if new_rate < 0.0:
            new_rate = 0.0
        elif new_rate > 1:
            new_rate = 1.0

        return new_rate


class KerasPackageWrapper:
    """Wraps around the ML package used - tf.Keras"""
    # TODO make such classes for tflearn and theano. Inherit them from a common abstract class.

    """Returns one hot encoding of the categorical matrix data in 2d numpy array"""
    @staticmethod
    def make_one_hot(categorical_data, n_categories):
        # TODO first
        incoding = keras.utils.to_categorical(categorical_data, num_classes=n_categories)

        return incoding

    """Returns tuple of the compiled 'model' trained on 'training_data' which is assumed normalized 
        and accuracy on the test data"""
    @staticmethod
    def do_training(model, x_train, y_train, x_test, y_test, n_categories, epochs):
        # TODO second
        one_hot_y_train_labels = KerasPackageWrapper.make_one_hot(y_train, n_categories)
        one_hot_y_test_labels = KerasPackageWrapper.make_one_hot(y_test, n_categories)

        # Keras model assumed
        trained_model = model.fit(x_train, one_hot_y_train_labels, epochs=epochs, batch_size=200)
        accuracy = KerasPackageWrapper.get_accuracy(trained_model, x_test, one_hot_y_test_labels)

        return trained_model, accuracy

    """Returns a sequential model with random architecture"""
    @staticmethod
    def make_flat_sequential_model():
        # TODO first

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

    """Copies the model"""
    @staticmethod
    def deep_copy(model):
        model.save("tempModelSave.h5")
        copied_model = keras.models.load_model("tempModelSave.h5", compile=True)

        return copied_model

    """Testing related"""
