import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from denspp.player.general_controller import DatasetReconstruction


class GeneralControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = DatasetReconstruction.__new__(DatasetReconstruction)
        self.controller._logger = MagicMock()


    def tearDown(self):
        del self.controller


    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_open_config_file_success(self, mock_yaml_load, mock_open):
        """Test successful loading of valid YAML config file"""
        expected_config = {"key": "value"}
        mock_yaml_load.return_value = expected_config
        self.controller._config_path = Path("/fake/path.yaml")

        result = self.controller._open_config_file()
        self.assertEqual(result, expected_config)
        mock_open.assert_called_once_with(self.controller._config_path.as_posix(), "r")


    @patch("builtins.open")
    def test_open_config_file_not_found(self, mock_open):
        """Test FileExistsError is raised if file is not found"""
        mock_open.side_effect = FileNotFoundError
        self.controller._config_path = Path("/nonexistent/path.yaml")

        with self.assertRaises(FileExistsError):
            self.controller._open_config_file()


    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_open_config_file_yaml_error(self, mock_yaml_load, mock_open):
        """Test yaml.YAMLError is raised when file parsing fails"""
        mock_yaml_load.side_effect = yaml.YAMLError("parsing error")
        self.controller._config_path = Path("/invalid/syntax.yaml")

        with self.assertRaises(yaml.YAMLError):
            self.controller._open_config_file()


    @patch("denspp.player.general_player_controller.DataTranslator")
    @patch("denspp.player.general_player_controller.hardware_specification_oscilloscope_mox4")
    def test_config_hardware_controller_success(self, mock_spec_mox4, mock_data_translator):
        """Test successful configuration with OscilloscopeMOX4"""
        self.controller._hardware_data_channel_mapping = ["CH1", "CH2"]
        self.controller._hardware_config_values = {
            "OscilloscopeMOX4": {"used": True},
            "OtherDevice": {"used": False},
        }
        mock_spec_instance = MagicMock()
        mock_spec_mox4.return_value = mock_spec_instance
        mock_translator_instance = MagicMock()
        mock_data_translator.return_value = mock_translator_instance

        result = self.controller._config_data_translator()
        self.assertEqual(result, mock_translator_instance)
        mock_spec_mox4.assert_called_once()
        mock_data_translator.assert_called_once_with(
            specific_device_settings=mock_spec_instance,
            logger=self.controller._logger,
            data_channel_mapping=self.controller._hardware_data_channel_mapping,
        )


    def test_config_hardware_controller_no_device_error(self):
        """Test error when no device is set to used"""
        self.controller._hardware_config_values = {
            "Device1": {"used": False},
            "Device2": {"used": False},
        }

        with self.assertRaisesRegex(Exception, "No hardware device is set to be used"):
            self.controller._config_data_translator()


    def test_config_hardware_controller_unknown_device_error(self):
        """Test error when an undefined device name is used"""
        unknown_device_name = "UnknownDeviceXYZ"
        self.controller._hardware_config_values = {unknown_device_name: {"used": True}}

        with self.assertRaisesRegex(
            Exception,
            f"The specified hardware device '{unknown_device_name}' is not defined",
        ):
            self.controller._config_data_translator()


if __name__ == "__main__":
    unittest.main()
