import os
from typing import Literal, Self

from pydantic import BaseModel

environment: dict[str, str] = dict(os.environ)


class Configuration(BaseModel):
    """Pydantic model representing application configuration."""

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARN"] = "INFO"


class ConfigurationService:
    """Class to instanciate."""

    __instance = None
    env: Configuration

    def __init__(self, env: Configuration) -> None:
        self.env = env

    @classmethod
    def get_instance(cls, conf: dict[str, str] = environment) -> Self:
        if cls.__instance is None:
            cls.__instance = cls(Configuration.model_validate(conf))

        return cls.__instance

    @classmethod
    def reset(cls) -> None:
        cls.__instance = None

    def get_dict(self) -> dict:
        return self.env.dict()
