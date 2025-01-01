from unittest import TestCase

from services.configuration_service import ConfigurationService


class TestConfigurationService(TestCase):
    """Test for ConfigurationService."""

    def setUp(self) -> None:
        ConfigurationService.reset()

    def test_get_instance(self) -> None:
        configuration: ConfigurationService = ConfigurationService.get_instance(
            {
                "LOG_LEVEL": "WARN",
            },
        )
        self.assertEqual("WARN", configuration.env.LOG_LEVEL)
