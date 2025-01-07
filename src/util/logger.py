from __future__ import annotations

import sys
import tensorflow as tf
import keras

import loguru
from loguru import logger

from services.configuration_service import ConfigurationService


def get_logger() -> loguru.Logger:
    """Configure logger for the configured log level.

    Returns:
        loguru.Logger: Configured logger

    """
    configuration = ConfigurationService.get_instance()

    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[ip]} {extra[user]} - <level>{message}</level>"
    )
    logger.configure(extra={"ip": "", "user": ""})
    logger.remove()
    logger.add(sys.stderr, format=logger_format, level=configuration.env.LOG_LEVEL)

    return logger


def startup(component: str) -> None:
    """Log app configuration configuration. Function to be called at app or component start.

    Args:
        component (str): Name of the app or component

    """
    logger = get_logger()
    logger.info(f"{component} starting")
    logger.info(f"Tensorflow version: {tf.__version__}")
    logger.info(f"Keras version: {keras.__version__}")
    configuration = ConfigurationService.get_instance().get_dict()

    for key in configuration:
        logger.info("{key}: {value}", key=key, value=configuration[key])
