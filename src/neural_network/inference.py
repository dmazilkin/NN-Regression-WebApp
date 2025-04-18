from keras.api.saving import load_model
from keras import Sequential

from helpers.config import TRAINED_MODEL

class NeuralNetwork:
    def __init__(self):
        # Load trained neural network
        self._nn: Sequential = load_model(TRAINED_MODEL)

    def predict(self, X):
        return self._nn.predict(X, verbose=0)

