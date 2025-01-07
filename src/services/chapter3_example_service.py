import keras
import tensorflow as tf
import urllib.request
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlparse
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


class Chapter3ExampleService:
    """Showcase example from Chapter 3. Going Beyond the Basics: Detecting Features in Images."""

    DATA_DIR = "data/chapter3/"
    COLOR_MODE_MONOCHROME = 1
    COLOR_MODE_RGB = 1

    # Conv2D will randomly initialize the amount of convolution filters, and over time will 
    # learn the filter values that work best to match the input values to their labels.
    MODEL_CONVOLUTION_LEARN = 64
    MODEL_CONVOLUTION_FILTER_SIZE = (3,3)

    def download_dataset(self, url: str, dest: str):
        file_name = "horse-or-human.zip"
        dest_dir = f"{self.DATA_DIR}{dest}"
        
        if not os.path.exists(dest_dir):
            try:
                get_logger().info(f"Downloading {url}")
                urllib.request.urlretrieve(url, file_name)

                get_logger().info(f"Extracting")
                zip_ref = zipfile.ZipFile(file_name, 'r')
                zip_ref.extractall(dest_dir)
                zip_ref.close()
            except:
                get_logger().warning(f"Failed downloading {url}")

        if os.path.exists(file_name):
            os.remove(file_name)

        get_logger().info(f"Dataset in {dest_dir}")

    def download_file(self, url: str) -> str:
        dest_dir = f"{self.DATA_DIR}testdata/"
        
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        filename = os.path.basename(urlparse(url).path)
        dest_filename = f"{dest_dir}{filename}"

        if not os.path.exists(dest_filename):
            get_logger().debug(f"Downloading {url}")
            (_path, _http_message) = urllib.request.urlretrieve(url, dest_filename)

        get_logger().info(f"File available {dest_filename}")

        return dest_filename
    
    def test_model(self, model: keras.Model, test_sample_url: str) -> None:
        path = self.download_file(test_sample_url)

        image = keras.preprocessing.image.load_img(path, target_size=(300,300))
        x = keras.preprocessing.image.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)

        if classes[0] > 0.5:
            get_logger().info(f"Human {classes[0]}")
        else:
            get_logger().info(f"Horse {classes[0]}")

    def save_model(self, model: keras.Model) -> None:
        if model.name is None:
            raise Exception("Model name must be asigned.")
        
        dest_dir = f"{self.DATA_DIR}models/"
        
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        
        filename = f"{dest_dir}{model.name}.keras"

        model.save(filename)

        get_logger().info(f"Saved model to {filename} (weights, architecture and optimizer state)")

    def load_model(self, name: str) -> keras.Model:
        filepath = f"{self.DATA_DIR}models/{name}.keras"

        model: keras.Model = keras.models.load_model(filepath, compile=True)

        return model

    def visualize(self, dataset, first: int = 9) -> None:
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(np.array(images[i]).astype("uint8"))
                #plt.title(int(labels[i]))
                plt.axis("off")        

    def example(self, callback: keras.callbacks.Callback | None = None) -> None:
        data = keras.datasets.fashion_mnist
        (training_images, training_labels), (test_images, test_labels) = data.load_data()

        training_images = training_images.reshape(60000, 28, 28, self.COLOR_MODE_MONOCHROME)
        training_images = training_images / 255.0
        test_images = test_images.reshape(10000, 28, 28, self.COLOR_MODE_MONOCHROME)
        test_images = test_images / 255.0

        model = keras.models.Sequential([
            # Convolutional filters
            keras.layers.Conv2D(
                self.MODEL_CONVOLUTION_LEARN, 
                self.MODEL_CONVOLUTION_FILTER_SIZE, 
                activation='relu', 
                input_shape=(28, 28, self.COLOR_MODE_MONOCHROME)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                self.MODEL_CONVOLUTION_LEARN, 
                self.MODEL_CONVOLUTION_FILTER_SIZE, 
                activation='relu'),
            keras.layers.MaxPooling2D(2,2),

            # Neural Network
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(
            training_images, 
            training_labels, 
            epochs=50,
            callbacks=[callback] if callback is not None else [])

        model.evaluate(test_images, test_labels)

        get_logger().info("Printing out model summary")
        model.summary()

        classifications = model.predict(test_images)
        print(classifications[0])
        print(test_labels[0])

        #get_logger().info(f"Evaluation against test data: {out}")

    def example_2(self, callback: keras.callbacks.Callback | None = None) -> None:
        get_logger().info(f"Keras version: {keras.version()}")

        self.download_dataset(
            url="https://storage.googleapis.com/learning-datasets/horse-or-human.zip",
            dest="horse-or-human/training/")

        training_data = keras.utils.image_dataset_from_directory(
            directory=f"{self.DATA_DIR}horse-or-human/training/",
            label_mode="binary",
            image_size=(300,300),
            batch_size=128,
        )

        for images, labels in training_data.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(np.array(images[i]).astype("uint8"))
                #plt.title(int(labels[i]))
                plt.axis("off")        

        self.download_dataset(
            url="https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip",
            dest="horse-or-human/validation/")

        validation_data = keras.utils.image_dataset_from_directory(
            directory=f"{self.DATA_DIR}horse-or-human/validation/",
            label_mode="binary",
            image_size=(300,300),
            batch_size=128,
        )

        model = keras.models.Sequential([
            keras.layers.Rescaling(scale=1./255),

            keras.layers.RandomFlip(mode="horizontal_and_vertical"),
            keras.layers.RandomZoom(height_factor=(-0.2, 0.2)),
            keras.layers.RandomRotation(factor=0.2),
            #keras.layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            #keras.layers.RandomContrast(factor=0.2),
            #keras.layers.RandomBrightness(factor=0.3),

            keras.layers.Conv2D(16, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ], name="humans_and_horses")

        model.compile(
            loss='binary_crossentropy',
            optimizer="rmsprop", # Would not accept optmizer instance although documentation says so
            metrics=['accuracy'])

        _history = model.fit(
            x=training_data,
            epochs=30,
            validation_data=validation_data,
            callbacks=[callback] if callback is not None else [])

        get_logger().info("Printing out model summary")
        model.summary()
        keras.utils.plot_model(model, show_shapes=True)

        self.save_model(model)
