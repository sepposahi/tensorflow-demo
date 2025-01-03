from dependency_injector import containers, providers

import services


class Container(containers.DeclarativeContainer):
    """Serves all application dependences."""

    linear_equation_service = providers.Singleton(services.LinearEquationService)
