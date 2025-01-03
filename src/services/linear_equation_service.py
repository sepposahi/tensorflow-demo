import keras
import numpy as np

from util.logger import get_logger


class LinearEquationService:
    """Showcase example from Chapter 1. Introduction to TensorFlow."""

    def temp(self) -> None:
        layer_dense = keras.layers.Dense(units=1, input_shape=[1])
        model = keras.Sequential()
        model.add(layer_dense)

        model.compile(optimizer="sgd", loss="mean_squared_error")

        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

        model.fit(xs, ys, epochs=500)

        prediction = model.predict(np.array([10.0], dtype=float))

        get_logger().info(f"Prediction: {prediction}")
        get_logger().info(f"Here is what I learned: {layer_dense.get_weights()}")
