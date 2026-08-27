from copy import deepcopy
from unittest import TestCase, main

import pytest
import torch

from denspp.offline import get_path_to_project
from denspp.offline.dnn.data_config import DefaultSettingsDataset, SettingsDataset
from denspp.offline.dnn.models.mnist import mnist_mlp_ae_v0, mnist_mlp_cl_v0
from denspp.offline.dnn.models.waveforms import waveforms_mlp_cl_v0

from .common_train import DefaultSettingsPytorch, PyTorchHandler, SettingsPytorch, TrainingsDevice


class TestCommonPyTorchTrain(TestCase):
    def setUp(self):
        set_data: SettingsDataset = deepcopy(DefaultSettingsDataset)
        set_data.data_type = "MNIST"
        set_train: SettingsPytorch = deepcopy(DefaultSettingsPytorch)
        self.dut = PyTorchHandler(config_train=set_train, config_dataset=set_data, do_train=True)

    def test_saving_path(self):
        rslt = self.dut.get_saving_path()
        self.assertEqual(rslt, get_path_to_project())

    def test_get_model(self):
        self.dut._settings_train.model_name = mnist_mlp_cl_v0.__name__
        rslt = self.dut._settings_train.get_model()
        self.assertEqual(type(rslt), mnist_mlp_cl_v0)

    def test_get_signature_mnist(self):
        self.dut._settings_train.model_name = mnist_mlp_cl_v0.__name__
        rslt = self.dut._settings_train.get_signature()
        self.assertEqual(rslt, [])

    def test_get_signature_waveform(self):
        self.dut._settings_train.model_name = waveforms_mlp_cl_v0.__name__
        rslt = self.dut._settings_train.get_signature()
        self.assertEqual(rslt, ["input_size", "output_size"])

    def test_model_number_parameters_non_defined(self):
        with self.assertRaises(AttributeError):
            self.dut.get_number_parameters_from_model

    def test_model_number_parameters_cl(self):
        model = mnist_mlp_cl_v0()
        self.dut.load_model(model=model)
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 31890)

    def test_model_number_parameters_ae(self):
        model = mnist_mlp_ae_v0()
        self.dut.load_model(model=model)
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 64554)

    def test_methods_custom_metrics(self):
        with self.assertRaises(AttributeError):
            self.dut.get_epoch_metric_custom_methods

    def test_define_ptq_level(self):
        self.dut.define_ptq_level(8, 5)


@pytest.mark.parametrize(
    "device, check",
    [
        (TrainingsDevice.auto, "auto"),
        (TrainingsDevice.cpu, "cpu"),
        (TrainingsDevice.cuda, "cuda"),
        (TrainingsDevice.mps, "mps"),
    ],
)
def test_settings_device_name(device: int, check: str) -> None:
    set_data: SettingsDataset = deepcopy(DefaultSettingsDataset)
    set_data.data_type = "MNIST"
    set_train: SettingsPytorch = deepcopy(DefaultSettingsPytorch)
    dut = PyTorchHandler(
        config_train=set_train, config_dataset=set_data, do_train=False, device_num=device
    )

    if device == TrainingsDevice.cuda:
        if not torch.cuda.is_available():
            try:
                dut._get_device_name
            except ValueError:
                pytest.skip("No CUDA device is available")
            else:
                assert dut._get_device_name == "cuda"
    elif device == TrainingsDevice.mps:
        if not torch.mps.is_available():
            try:
                dut._get_device_name
            except ValueError:
                pytest.skip("No MPS device is available")
            else:
                assert dut._get_device_name == "mps"
    else:
        assert dut._get_device_name == check


@pytest.mark.parametrize(
    "device, check",
    [
        (TrainingsDevice.auto, "auto"),
        (TrainingsDevice.cpu, "cpu"),
    ],
)
def test_settings_device_type(device: int, check: str) -> None:
    set_data: SettingsDataset = deepcopy(DefaultSettingsDataset)
    set_data.data_type = "MNIST"
    set_train: SettingsPytorch = deepcopy(DefaultSettingsPytorch)
    dut = PyTorchHandler(
        config_train=set_train, config_dataset=set_data, do_train=False, device_num=device
    )

    if device == TrainingsDevice.auto:
        assert dut.get_device != check
    else:
        assert dut.get_device == check


if __name__ == "__main__":
    main()
