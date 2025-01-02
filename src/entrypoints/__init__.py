import tensorflow as tf

from util.logger import get_logger, startup

component = "tensorflow-demo"


def main() -> None:
    """Command line entrypoint."""
    startup(component)


def version() -> None:
    """Command line entrypoint that displays Tensorflow version."""
    startup(component)
    get_logger().info(tf.__version__)
