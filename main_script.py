from tensorflow import keras

"""The main class for the optimizer"""
class GeneticOptimizer:
    def __init__(self, base_model, test_data, n_parents=2, traits=dict(), n_iterations=100):
        # TODO add assertion about the traits in relation to the attributes of the model
        self.test_data = test_data
        self.n_parents = n_parents
        self.base_model = base_model
        self.traits = traits  # Hyper parameters to optimize
        self.n_iterations = n_iterations
        self.optimized_models = self.generate_population(base_model)  # A sorted list models (by performance) after the lest evolutionary step

    """Breeds the parents (already sorted models) and returns a list containing the parents and 
    the mutated children as compiled models."""
    def breed(self, models, n_parents):
        # TODO second
        assert n_parents <= len(models)

        INVARIANT_MODELS = models[:n_parents]  # parents - will remain unchanged
        n_children = len(models) - n_parents
        children = []

        for i in range(n_children):
            pure_child = self.inherit_to_child(INVARIANT_MODELS, self.traits)
            mutated_child = self.mutate(pure_child)

            children.append(mutated_child)

        return INVARIANT_MODELS.concat(children)

    """Returns a list of models where for every compiled model from the 'models' there is a tuple of the trained model
    and the accuracy on test data"""
    def train_models(self, models):
        # TODO second
        trained_models = models[:]

        for model in models:
            trained_model, accuracy = self.do_training(model)
            trained_models.append((trained_model, accuracy))

        return trained_models

    """Returns a list of models with the same architecture as 'base_model' but optimal hyper-parameters"""
    def optimize(self):
        # TODO third
        # Let the evolution run for n_iterations
        for i in range(self.n_iterations):
            models = self.breed(self.optimized_models, self.n_parents)
            trained_models = self.train_models(models)  # A list of (trained model, accuracy) tuples
            self.optimized_models = sorted(trained_models, key=lambda t1, t2: t2)

    """Returns a list of compiled models with randomized hyper-parameters and the same architecture as the 'model'"""
    def generate_population(self, model):
        # TODO first
        return keras.Sequential()

    """Returns a child with a random set of traits from its parents"""
    def inherit_to_child(self, parents, traits):
        # TODO first
        return keras.Sequential()

    """Returns the same model but with slightly altered hyper-parameters"""
    def mutate(self, child):
        # TODO first
        return child

    """Returns the trained version of the compiled 'model'"""
    def do_training(self, model):
        # TODO first
        accuracy = 0

        return model, accuracy
