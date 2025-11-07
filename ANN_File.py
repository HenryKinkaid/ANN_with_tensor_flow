import tensorflow as tf
import keras
from keras import layers

class ANN():
    def __init__(self,train, test):
        # define the model with keras
        train_data = train
        test_data = test

        self.model = keras.Sequential(
            [
                layers.Dense(6, activation="relu", name="layer1"),
                layers.Dense(6, activation="relu", name="layer2"),
                layers.Dense(6, activation="relu", name="layer3"),
                layers.Dense(2, name="layer4"),
            ]
        )

    def test_model(self, inputs: list):
        return self.model(inputs)

    def train_model_n_epochs(self, n):
        pass