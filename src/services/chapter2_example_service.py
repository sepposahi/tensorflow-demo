import keras
import tensorflow as tf

from util.logger import get_logger


class StopTrainingCallback(keras.callbacks.Callback):
    """Keras Callback to stop training."""

    def __init__(self, desired_accuracy: float = 0.95) -> None:
        self.desired_accuracy = desired_accuracy

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:  # noqa: ARG002
        accuracy = logs.get("accuracy") if logs else None
        if accuracy and accuracy > self.desired_accuracy and self.model is not None:
            get_logger().info(
                f"Reached desired accuracy so cancelling training (epoch = {epoch}, accuracy = {accuracy})"
            )
            self.model.stop_training = True


class Chapter2ExampleService:
    """Showcase example from Chapter 2. Neurons for Vision."""

    def example(self, callback: keras.callbacks.Callback | None = None) -> None:
        data = keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = data.load_data()

        """
        Python allows you to do an operation across the entire array with this notation. Recall
        that all of the pixels in our images are grayscale, with values between 0 and 255. Dividing
        by 255 thus ensures that every pixel is represented by a number between 0 and 1 instead.
        This process is called normalizing the image.
        """
        training_images = training_images / 255.0
        test_images = test_images / 255.0

        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(training_images, training_labels, epochs=50, callbacks=[callback] if callback is not None else [])

        out = model.evaluate(test_images, test_labels, return_dict=True)

        get_logger().info(f"Evaluation against test data: {out}")
