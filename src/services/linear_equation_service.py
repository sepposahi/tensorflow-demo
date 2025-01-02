class LinearEquationService:
    """Showcase example from Chapter 1. Introduction to TensorFlow."""

    __instance = None

    def __init__(self) -> None:
        if self.__instance is None:
            self.__instance = self
