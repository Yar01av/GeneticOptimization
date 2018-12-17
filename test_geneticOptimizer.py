from unittest import TestCase
from main_script import GeneticOptimizer, KerasPackageWrapper
import numpy as np
import keras
from testing_data import get_clean_mnist, get_clean_mnist_with_cold_labels


class TestGeneticOptimizer(TestCase):
    # Clean mnist data. Ready to use by the networks
    train_data_x, train_data_y, test_data_x, test_data_y = get_clean_mnist()

    # Like assertAlmost but for multiple numbers
    def _assertAlmostEqualsMultiple(self, tested_value, values_to_compare, delta):
        # Check if there are any values that are close enough
        filter(lambda x: abs(x - tested_value) <= delta, values_to_compare)

        if len(values_to_compare) == 0:
            self.fail("Some values_to_compare are too far away from the tested_value")

    # Extracts rates of the list of models and returns them as a list
    @staticmethod
    def _extract_rates(models, layer):
        return [model.layers[layer].rate for model in models]

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
        self.assertAlmostEqual(model.layers[1].rate, child.layers[1].rate, delta=0.2)
        self.assertEqual(model.layers[2].units, child.layers[2].units)
        self.assertAlmostEqual(model.layers[3].rate, child.layers[3].rate, delta=0.2)

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

        parents = [model1, model2]
        child = GeneticOptimizer.inherit_to_child(parents, dict(layer_dropout={1}), delta)
        # Assert that the structure stays the same as that of parents but the rate changes
        self.assertEqual(child.layers[0].units, model1.layers[0].units)
        self.assertEqual(child.layers[0].units, model2.layers[0].units)

        self._assertAlmostEqualsMultiple(child.layers[1].rate, TestGeneticOptimizer._extract_rates(parents, 1), delta)

    # TODO make test_inherit_to_child4 that would not be passed by multiplexer

    """Test train_models (non-assertive)"""

    def test_train_models1(self):
        # TODO keras assumed!
        model1 = KerasPackageWrapper.make_flat_sequential_model()
        model1.add(keras.layers.Dense(300, activation="relu", input_dim=784))
        model1.add(keras.layers.Dropout(0.2))
        model1.add(keras.layers.Dense(10, activation="softmax"))
        model1.add(keras.layers.Dropout(0.2))

        model1.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        instance = GeneticOptimizer(model1, (get_clean_mnist_with_cold_labels()[:2]),
                                    (get_clean_mnist_with_cold_labels()[2:]), n_categories=10,
                                    traits=dict(layer_dropout={1, 3}))

        print(GeneticOptimizer.train_models(instance, [model1]))

    def test_train_models2(self):
        model1 = KerasPackageWrapper.make_flat_sequential_model()
        model1.add(keras.layers.Dense(300, activation="relu", input_dim=784))
        model1.add(keras.layers.Dropout(0.2))
        model1.add(keras.layers.Dense(10, activation="softmax"))
        model1.add(keras.layers.Dropout(0.2))

        model2 = KerasPackageWrapper.make_flat_sequential_model()
        model2.add(keras.layers.Dense(300, activation="relu", input_dim=784))
        model2.add(keras.layers.Dropout(0.5))
        model2.add(keras.layers.Dense(10, activation="softmax"))
        model2.add(keras.layers.Dropout(0.5))

        model3 = KerasPackageWrapper.make_flat_sequential_model()
        model3.add(keras.layers.Dense(300, activation="relu", input_dim=784))
        model3.add(keras.layers.Dropout(0.7))
        model3.add(keras.layers.Dense(10, activation="softmax"))
        model3.add(keras.layers.Dropout(0.7))

        model1.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        model2.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        model3.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

        instance = GeneticOptimizer(model1, (get_clean_mnist_with_cold_labels()[:2]),
                                    (get_clean_mnist_with_cold_labels()[2:]), n_categories=10,
                                    traits=dict(layer_dropout={1, 3}))

        print(GeneticOptimizer.train_models(instance, [model1, model2, model3]))

    """Tests breed"""

    def test_breed1(self):
        delta = 0.0
        n_parents = 1
        traits = dict(layer_dropout={1})

        model_p = KerasPackageWrapper.make_flat_sequential_model()
        model_p.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_p.add(keras.layers.Dropout(0.1))

        model_c1 = KerasPackageWrapper.make_flat_sequential_model()
        model_c1.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c1.add(keras.layers.Dropout(0.3))

        model_c2 = KerasPackageWrapper.make_flat_sequential_model()
        model_c2.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c2.add(keras.layers.Dropout(0.7))

        new_generation = GeneticOptimizer.breed([model_p, model_c1, model_c2], n_parents,
                                                traits, delta)

        for layer_i in traits["layer_dropout"]:
            self.assertEqual(model_p.layers[layer_i - 1].units, new_generation[1].layers[layer_i - 1].units)
            self.assertEqual(model_p.layers[layer_i].rate, new_generation[1][layer_i].layers[1].rate)

            self.assertEqual(model_p.layers[layer_i - 1].units, new_generation[2].layers[layer_i - 1].units)
            self.assertEqual(model_p.layers[layer_i].rate, new_generation[2][layer_i].layers[1].rate)

    def test_breed2(self):
        delta = 0.3
        n_parents = 1
        traits = dict(layer_dropout={1})

        model_p = KerasPackageWrapper.make_flat_sequential_model()
        model_p.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_p.add(keras.layers.Dropout(0.1))

        model_c1 = KerasPackageWrapper.make_flat_sequential_model()
        model_c1.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c1.add(keras.layers.Dropout(0.3))

        model_c2 = KerasPackageWrapper.make_flat_sequential_model()
        model_c2.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c2.add(keras.layers.Dropout(0.7))

        new_generation = GeneticOptimizer.breed([model_p, model_c1, model_c2], n_parents,
                                                traits, delta)

        for layer_i in traits["layer_dropout"]:
            for network in new_generation[n_parents:]:
                self.assertEquals(model_p.layers[layer_i - 1].units, network.layers[layer_i - 1].units)
                self.assertAlmostEqual(model_p.layers[layer_i].rate, network.layers[1].rate, delta)

    def test_breed3(self):
        delta = 0.3
        n_parents = 2
        traits = dict(layer_dropout={1})

        model_p = KerasPackageWrapper.make_flat_sequential_model()
        model_p.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_p.add(keras.layers.Dropout(0.1))

        model_c1 = KerasPackageWrapper.make_flat_sequential_model()
        model_c1.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c1.add(keras.layers.Dropout(0.3))

        model_c2 = KerasPackageWrapper.make_flat_sequential_model()
        model_c2.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c2.add(keras.layers.Dropout(0.7))

        model_c3 = KerasPackageWrapper.make_flat_sequential_model()
        model_c3.add(keras.layers.Dense(10, activation="relu", input_dim=10))
        model_c3.add(keras.layers.Dropout(0.9))

        new_generation = GeneticOptimizer.breed([(model_p, 0.5), (model_c1, 0.3), (model_c2, 0.2), (model_c3, 0.1)],
                                                n_parents, traits, delta)

        for layer_i in traits["layer_dropout"]:
            for network in new_generation[n_parents:]:
                self.assertEquals(model_p.layers[layer_i - 1].units, network.layers[layer_i - 1].units)

                self._assertAlmostEqualsMultiple(network.layers[layer_i].rate,
                                                 TestGeneticOptimizer._extract_rates(new_generation[:n_parents], layer_i),
                                                 delta)
