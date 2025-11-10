import tensorflow as tf
import keras
from keras import layers

class ANN():
    def __init__(self):
        # define the model with keras

        self.model = keras.Sequential(
            [
                layers.Dense(6, activation="relu", name="layer1"),
                layers.Dense(6, activation="relu", name="layer2"),
                layers.Dense(6, activation="relu", name="layer3"),
                layers.Dense(2, name="layer4"),
            ]
        )

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                           metrics=["mae"])

    def test_model(self, inputs: list):
        return self.model.predict(inputs)

    def train_model_n_epochs(self, n, batch_size, inputs, outputs):
        self.model.fit(x=inputs,
                     y=outputs,
                     batch_size=batch_size,
                     epochs=n,
                     validation_split=0.2,
                     verbose=1)