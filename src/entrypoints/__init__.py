import tensorflow as tf

from containers import Container
from services.chapter2_example_service import StopTrainingCallback
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


def chapter2() -> None:
    """Entrypoint that runs Chapter 2 example code."""
    startup(component)
    service = container.chapter2_example_service()
    service.example(callback=StopTrainingCallback())

def chapter3() -> None:
    """Entrypoint that runs Chapter 3 example code."""
    startup(component)
    service = container.chapter3_example_service()
    service.example(callback=StopTrainingCallback(desired_accuracy=0.8))

def chapter3_2() -> None:
    """Entrypoint that runs Chapter 3 example code with Horses and Humans dataset."""
    startup(component)
    service = container.chapter3_example_service()
    service.example_2(callback=StopTrainingCallback(desired_accuracy=0.9))

    model = service.load_model("humans_and_horses")

    # Horses
    horses = [
        "https://www.365vet.co.uk/media/magefan_blog/Horse_close_up.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Roan.jpg/640px-Roan.jpg",
        "https://www.foxdenequine.com/cdn/shop/articles/Horse_in_Pasture.jpg",
        "https://www.thesprucepets.com/thmb/WCHi1vABy_vfH6kDo7snIluUzyo=/2109x0/filters:no_upscale():strip_icc()/GettyImages-909948608-5c69cd9446e0fb0001560d1a.jpg",
        "https://i.natgeofe.com/n/9957ad43-61ef-45a4-9900-98e2581a0f3b/przewalskis-horse_thumb_2x3.JPG",
    ]

    for horse in horses:
        service.test_model(
            model=model, 
            test_sample_url=horse)

    # Humans
    humans = [
        "https://thumbs.dreamstime.com/z/black-man-running-sprinting-beach-motivated-city-athlete-training-hard-outdoor-77347606.jpg",
        "https://images.ctfassets.net/6ilvqec50fal/7KJ4F4dd4XnHVtL8BMmjOC/02db68080e36ee8af409b1198df26ce7/Cross-training-for-runners.jpg",
        "https://as2.ftcdn.net/v2/jpg/02/38/45/27/1000_F_238452704_ZKlnIKazn4KxagIoU2qLA9P1iT0U4SnT.jpg",
        "https://americanhiking.org/wp-content/uploads/2015/03/lizsocal.jpg",
        "https://snaped.fns.usda.gov/sites/default/files/gallery/2018-04/activity_vertical3.jpg"
    ]

    for human in humans:
        service.test_model(
            model=model, 
            test_sample_url=human)
