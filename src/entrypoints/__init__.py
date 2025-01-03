import tensorflow as tf

from containers import Container
from util.logger import get_logger, startup

component = "tensorflow-demo"

container = Container()
container.wire(modules=["services.linear_equation_service"])


def version() -> None:
    """Command line entrypoint that displays Tensorflow version."""
    startup(component)
    get_logger().info(tf.__version__)


def linear_equation() -> None:
    """Command line entrypoint that displays Tensorflow version."""
    startup(component)
    linear_equation_service = container.linear_equation_service()

    linear_equation_service.temp()
