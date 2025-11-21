import tensorflow as tf
import keras
from keras import layers
import numpy as np
THRESHOLD = .0005

class ANN():
    def __init__(self, filepath=None):
        # define the model with keras
        if filepath is None:
            self.model = keras.Sequential(
                [
                    layers.Dense(128, activation="relu", name="layer1"),
                    layers.Dense(64, activation="relu", name="layer2"),
                    layers.Dense(32, activation="relu", name="layer3"),
                    layers.Dense(2, name="layer4"),
                ]
            )

            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                               metrics=["mae"])

        else:
            self.model = tf.keras.models.load_model(filepath)

    def test_model(self, inputs):
        return self.model.predict(inputs)

    def train_model_n_epochs(self, n, batch_size, inputs, outputs):
        history = self.model.fit(x=inputs,
                     y=outputs,
                     batch_size=batch_size,
                     epochs=n,
                     validation_split=0.2,
                     verbose=1,
                    shuffle=True)
        return history

    def create_full_run(self, inputs, length=100):
        print(f"inputs.shape: {inputs.shape}")
        run = inputs # note that run is a 2d array with a shape (1, n) because this is the array shape expected by tensorflow
        for i in range(length):

            next_point = self.model.predict(inputs)
            # print(next_point[0][0] - run[0][-2])
            if abs(next_point[0][0] - run[0][-2]) < THRESHOLD and abs(next_point[0][1] - run[0][-1]) < THRESHOLD:
                print(i)
                return run
            run = np.append(run[0], next_point[0][0]) # run is a 2d array, so run[0] returns the actual list of points
            run = np.append(run, next_point[0][1]) # at this point, run is no longer a 2d array, so run itself is just the list of points
            run = np.expand_dims(run, axis=0) # turns run back into a 2d array

            inputs = run[:, -6:]
            # print(inputs.shape)
        return run
