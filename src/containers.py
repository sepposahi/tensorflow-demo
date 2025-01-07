from dependency_injector import containers, providers

import services


class Container(containers.DeclarativeContainer):
    """Serves all application dependences."""

    linear_equation_service = providers.Singleton(services.LinearEquationService)
    chapter2_example_service = providers.Singleton(services.Chapter2ExampleService)
    chapter3_example_service = providers.Singleton(services.Chapter3ExampleService)
