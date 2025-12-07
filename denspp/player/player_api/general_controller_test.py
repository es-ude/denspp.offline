import unittest
from unittest.mock import patch, Mock, mock_open, MagicMock
import inspect
import yaml

import denspp.player.src.general_player_controller as general_player_controller
import hardware_settings

class GeneralControllerTest(unittest.TestCase):
    def setUp(self):
        # Setup code before each test
        self.controller = general_player_controller.General_controller()
        self.controller.logger = Mock()
        self.controller.config_path = "test/path"


    def tearDown(self):
        # Cleanup code after each test
        del self.controller


    
# - - - -  Test for the "_read_sys_args" methode - - - - 
    @patch("general_controller.sys.argv", ["general_controller.py", "test/path"])
    def test_read_sys_args_with_argument(self):
        """Test reading sys.argv with an argument provided."""
        expected_path = "test/path"
        result_path = self.controller._read_sys_args()
        self.assertEqual(result_path, expected_path)
    
    @patch("general_controller.sys.argv", ["general_controller.py"])
    def test_read_sys_args_without_argument(self):
        """Test reading sys.argv without any argument provided."""
        expected_path = "2_gui/hardware_config.yaml"
        result_path = self.controller._read_sys_args()
        self.assertEqual(result_path, expected_path)
    

# - - - -  Test for the "_open_config_file" methode - - - -
    def test_open_config_file_success(self):
        """Test opening and reading a valid config file."""
        mock_yaml_content = mock_data_for_yaml()
        except_dict = mock_data_for_general_config()
        with patch('builtins.open', mock_open(read_data=mock_yaml_content)) as mock_file, \
            patch('general_controller.sys.exit') as mock_exit:
            result = self.controller._open_config_file()

            mock_file.assert_called_once_with(f'{self.controller.config_path}', 'r')
            self.assertEqual(result, except_dict)
            self.controller.logger.info.assert_called_with("Configuration file loaded successfully")
            mock_exit.assert_not_called()

    @patch('general_controller.sys.exit', side_effect=SystemExit)
    @patch('builtins.open', side_effect=FileNotFoundError) 
    def test_open_config_file_not_found(self, mock_open, mock_exit):
        """Test opening a config file that does not exist."""
        with self.assertRaises(SystemExit):
            result = self.controller._open_config_file()
        self.controller.logger.critical.assert_called_with(
            f"The File/Path '{self.controller.config_path}' are not found"
        )
        mock_exit.assert_called_once()

    @patch('general_controller.sys.exit', side_effect=SystemExit)
    @patch('general_controller.yaml.safe_load', side_effect=yaml.YAMLError("Bad YAML"))
    @patch('builtins.open', new_callable=lambda: mock_open(read_data="bad: data:")) 
    def test_open_config_yaml_error(self, mock_open, mock_safe_load, mock_exit):
        """Test opening a config file with invalid YAML content."""
        with self.assertRaises(SystemExit):
            self.controller._open_config_file()
        self.controller.logger.critical.assert_called_with(
            "Error parsing the YAML file: Bad YAML"
        )
        mock_exit.assert_called_once()


# - - - -  Test for the "_read_config" methode - - - -
    def test_read_config(self):
        """Test reading and processing the configuration settings."""
        self.controller.general_config = mock_data_for_general_config()
        self.controller._read_config()
        control_values = mock_data_for_general_config()
    
        self.assertEqual(self.controller.logging_lvl, control_values['General_Configuration']['Logging_Lvl'])
        self.assertEqual(self.controller.data_path, control_values['Data_Input']['input'])
        self.assertEqual(self.controller.hardware_config_values , control_values['Hardware'])
        self.assertEqual(self.controller.data_config_values, control_values['Data_Configuration'])
        self.assertEqual(self.controller.data_config_do_cut, control_values['Data_Configuration']['Data_Preprocessing']['do_cut'])
        self.assertEqual(self.controller.data_config_do_resample, control_values['Data_Configuration']['Data_Preprocessing']['do_resampling'])
    

# - - - -  Test for the "_config_hardware_controller" methode - - - -
    @patch('general_controller.Hardware_settings', spec=hardware_settings.Hardware_settings)
    @patch('general_controller.output_devices')
    @patch('general_controller.inspect.getmembers')
    @patch('general_controller.sys.exit', side_effect=SystemExit)
    def test_config_hardware_controller_finds_device(self, mock_exit, mock_getmembers, mock_output_devices, MockHardwareSettings):
        """Testing _config_hardware_controller when a valid device is selected"""
        DEVICE_NAME = 'MockDevice'

        self.controller.hardware_config_values = {
            DEVICE_NAME: {'used': True, 'output_open': True},
            'AnotherDevice': {'used': False}
        }

        mock_output_devices.__name__ = 'mocked_output_devices_module'

        fake_validation_class = MagicMock(name="ValidationClass")
        fake_validation_class.__module__ = 'mocked_output_devices_module'
        mock_getmembers.return_value = [(DEVICE_NAME, fake_validation_class)]

        fake_specific_class = MagicMock(name="SpecificClass")
        fake_specific_class_instance = Mock(name="Instance")
        fake_specific_class.return_value = fake_specific_class_instance
        setattr(mock_output_devices, DEVICE_NAME, fake_specific_class)

        self.controller._config_hardware_controller()
        
        mock_getmembers.assert_called_once_with(mock_output_devices, inspect.isclass)
        MockHardwareSettings.assert_called_once_with(
            fake_specific_class_instance, self.controller.logger
        )

        self.assertEqual(fake_specific_class_instance.output_open, True)
        mock_exit.assert_not_called()

    @patch('general_controller.sys.exit', side_effect=SystemExit)
    def test__config_hardware_controller_no_device_selected(self, mock_exit):
        """Tests whether sys.exit() is called when no device has "used: True"""
        self.controller.hardware_config_values = {
            'DeviceA': {'used': False},
            'DeviceB': {'used': False}
        }
        with self.assertRaises(SystemExit):
            self.controller._config_hardware_controller()
        self.controller.logger.CRITICAL.assert_called_with(
            "No hardware device is set to be used. Please select one device."
        )
        mock_exit.assert_called_once()

    @patch('general_controller.sys.exit', side_effect=SystemExit)
    def test__config_hardware_controller_multiple_devices_selected(self, mock_exit):
        """Tests whether sys.exit() is called when multiple devices have 'used: True'"""
        self.controller.hardware_config_values = {
            'DeviceA': {'used': True},
            'DeviceB': {'used': True}
        }
        with self.assertRaises(SystemExit):
            self.controller._config_hardware_controller()
        self.controller.logger.CRITICAL.assert_called_with(
            "Multiple hardware devices are set to be used. Please select only one device."
        )
        mock_exit.assert_called_once()

    @patch('general_controller.output_devices')
    @patch('general_controller.inspect.getmembers')
    @patch('general_controller.sys.exit', side_effect=SystemExit)
    def test__config_hardware_controller_device_not_found(self, mock_exit, mock_getmembers, mock_output_devices):
        """Tests whether sys.exit() is called when the 'used' device is not found in output_devices.py."""
        DEVICE_NAME = 'MissingDevice'
        self.controller.hardware_config_values = {
            DEVICE_NAME: {'used': True}
        }
        mock_output_devices.__name__ = 'mocked_output_devices_module'
        fake_other_class = MagicMock(name="OtherClass")
        fake_other_class.__module__ = 'mocked_output_devices_module'
        
        mock_getmembers.return_value = [
            ('SomeOtherDevice', fake_other_class) 
        ]
        with self.assertRaises(SystemExit):
            self.controller._config_hardware_controller()
        mock_getmembers.assert_called_once_with(mock_output_devices, inspect.isclass)
        expected_log_tuple = (DEVICE_NAME, {'used': True})
        self.controller.logger.CRITICAL.assert_called_with(
            f"The specified hardware device '{expected_log_tuple}' is not defined in output_devices.py."
        )
        mock_exit.assert_called_once()


# - - - -  Test for the "_config_DataController" methode - - - -
    @patch('general_controller.DataController')
    def test_config_DataController_with_time_range(self, MockDataController):
        """Test _config_DataController when start_time and end_time are both not None"""
        mock_dc_instance = Mock()
        MockDataController.return_value = mock_dc_instance
        
        self.controller.data_path = "test/data/path"
        self.controller.data_config_values = {
            'Time': {
                'start_time': 1.0,
                'end_time': 5.0
            },
            'Data_Preprocessing': {
                'channel_selection': [0, 1, 2],
                'sampling_rate_resample': 1000
            }
        }
        
        self.controller._config_DataController()
        
        MockDataController.assert_called_once()
        self.assertEqual(self.controller.deployed_data_controller, mock_dc_instance)
        self.assertEqual(mock_dc_instance.logger, self.controller.logger)
        self.assertEqual(mock_dc_instance.path, "test/data/path")
        self.assertEqual(mock_dc_instance.t_range, [1.0, 5.0])
        self.assertEqual(mock_dc_instance.ch_sel, [0, 1, 2])
        self.assertEqual(mock_dc_instance.fs_resample, 1000)

    @patch('general_controller.DataController')
    def test_config_DataController_without_time_range(self, MockDataController):
        """Test _config_DataController when start_time or end_time is None"""
        mock_dc_instance = Mock()
        MockDataController.return_value = mock_dc_instance
        
        self.controller.data_path = "test/data/path2"
        self.controller.data_config_values = {
            'Time': {
                'start_time': None,
                'end_time': 5.0
            },
            'Data_Preprocessing': {
                'channel_selection': [],
                'sampling_rate_resample': 500
            }
        }
        
        self.controller._config_DataController()
        
        MockDataController.assert_called_once()
        self.assertEqual(self.controller.deployed_data_controller, mock_dc_instance)
        self.assertEqual(mock_dc_instance.logger, self.controller.logger)
        self.assertEqual(mock_dc_instance.path, "test/data/path2")
        self.assertEqual(mock_dc_instance.t_range, [])
        self.assertEqual(mock_dc_instance.ch_sel, [])
        self.assertEqual(mock_dc_instance.fs_resample, 500)

    @patch('general_controller.DataController')
    def test_config_DataController_both_times_none(self, MockDataController):
        """Test _config_DataController when both start_time and end_time are None"""
        mock_dc_instance = Mock()
        MockDataController.return_value = mock_dc_instance
        
        self.controller.data_path = "test/data/path3"
        self.controller.data_config_values = {
            'Time': {
                'start_time': None,
                'end_time': None
            },
            'Data_Preprocessing': {
                'channel_selection': [1],
                'sampling_rate_resample': 2000
            }
        }
        
        self.controller._config_DataController()
        
        MockDataController.assert_called_once()
        self.assertEqual(mock_dc_instance.t_range, [])
        self.assertEqual(mock_dc_instance.ch_sel, [1])
        self.assertEqual(mock_dc_instance.fs_resample, 2000)


# - - - -  Test for the "_config_board_dataset" methode - - - -
    @patch('general_controller.Board_dataset')
    def test_config_board_dataset_with_labels(self, MockBoardDataset):
        """Test _config_board_dataset when labels exist in the data controller"""
        mock_board_instance = Mock(spec=hardware_settings.Board_dataset)
        mock_board_instance.samplingrate = None
        mock_board_instance.groundtruth = []

        MockBoardDataset.return_value = mock_board_instance
        
        mock_data_controller = Mock()
        mock_data_controller.data_fs_current = 1000
        mock_data_controller.label_exist = True
        mock_data_controller.spike_xpos = [10, 20, 30]
        
        self.controller.deployed_data_controller = mock_data_controller
        
        self.controller._config_board_dataset()
        
        MockBoardDataset.assert_called_once()
        self.assertEqual(self.controller.deployed_board_dataset, mock_board_instance)
        self.assertEqual(mock_board_instance.samplingrate, 1000)
        self.assertEqual(mock_board_instance.groundtruth, [10, 20, 30])

    @patch('general_controller.Board_dataset', spec=hardware_settings.Board_dataset)
    def test_config_board_dataset_without_labels(self, MockBoardDataset):
        """Test _config_board_dataset when labels do not exist in the data controller"""

        mock_board_instance = Mock(spec=hardware_settings.Board_dataset)
        mock_board_instance.samplingrate = None
        mock_board_instance.groundtruth = []
        
        MockBoardDataset.return_value = mock_board_instance
        
        mock_data_controller = Mock()
        mock_data_controller.data_fs_current = 2000
        mock_data_controller.label_exist = False
        
        self.controller.deployed_data_controller = mock_data_controller
        
        self.controller._config_board_dataset()
        
        MockBoardDataset.assert_called_once()
        self.assertEqual(self.controller.deployed_board_dataset, mock_board_instance)
        self.assertEqual(mock_board_instance.samplingrate, 2000)
        self.assertEqual(mock_board_instance.groundtruth, [])




# - - - - Helper Functions - - - -

def mock_data_for_yaml():
    mock_yaml ="""
        General_Configuration:
            Logging_Lvl: "DEBUG"
        Data_Input:
            input: "Test_Waveform"
        Hardware:
            OscilloscopeMOX4:
                used: True
                output_open: False
            Oscilloscope3000:
                used: False
        Data_Configuration:
            Time:
                start_time: 1
                end_time: 2
            Data_Preprocessing:
                channel_selection: []
                do_cut: True
                do_resampling: True
                sampling_rate_resample: 999
        """
    return mock_yaml

def mock_data_for_general_config():
    return{
            'General_Configuration': {
                'Logging_Lvl': "DEBUG"
            },
            'Data_Input': {
                'input': "Test_Waveform"
            },
            'Hardware': {
                'OscilloscopeMOX4': {
                    'used': True,
                    'output_open': False
                },
                'Oscilloscope3000': {
                    'used': False
                }
            },
            'Data_Configuration': {
                'Time': {
                    'start_time': 1,
                    'end_time': 2
                },
                'Data_Preprocessing': {
                    'channel_selection': [],
                    'do_cut': True,
                    'do_resampling': True,
                    'sampling_rate_resample': 999
                }
            }}

if __name__ == '__main__':
    unittest.main()