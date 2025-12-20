from copy import deepcopy
from unittest import TestCase, main
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam

from denspp.offline.dnn.data_config import (
    SettingsDataset,
    DefaultSettingsDataset
)
from denspp.offline.dnn.models.mnist import mnist_mlp_cl_v0, mnist_mlp_ae_v0
from .common_train import (
    SettingsPytorch,
    DefaultSettingsTrainMSE,
    DefaultSettingsTrainCE,
    PyTorchHandler
)

# --- Info: Function have to start with test_*
class TestPyTorchModelConfigClassifier(TestCase):
    def setUp(self):
        self.sets: SettingsPytorch = deepcopy(DefaultSettingsTrainCE)

    def test_get_model_overview(self):
        rslt = self.sets.get_model_overview(print_overview=True)
        assert len(rslt) > 0
        assert 'mnist_mlp_cl_v0' in rslt
        assert 'waveforms_mlp_cl_v0' in rslt

    def test_no_model_defined(self):
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_wrong_model_mnist(self):
        self.sets.model_name = 'mnist_test_cl_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_model_mnist(self):
        self.sets.model_name = 'mnist_test_cl_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_model_waveforms(self):
        self.sets.model_name = 'waveforms_mlp_cl_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_get_loss_func(self):
        rslt = self.sets.get_loss_func()
        assert type(rslt) == CrossEntropyLoss

    def test_load_optimizer(self):
        self.sets.model_name = 'waveforms_mlp_cl_v0'
        model = self.sets.get_model()
        rslt = self.sets.load_optimizer(
            model=model,
            learn_rate=0.2
        )
        assert type(rslt) == Adam


class TestPyTorchModelConfigAutoencoder(TestCase):
    def setUp(self):
        self.sets: SettingsPytorch = deepcopy(DefaultSettingsTrainMSE)

    def test_get_model_overview(self):
        rslt = self.sets.get_model_overview(print_overview=True)
        assert len(rslt) > 0
        assert 'mnist_mlp_ae_v0' in rslt
        assert 'waveforms_mlp_ae_v0' in rslt

    def test_no_model_defined(self):
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_wrong_model_mnist(self):
        self.sets.model_name = 'mnist_mlp_ae_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_model_mnist(self):
        self.sets.model_name = 'mnist_mlp_ae_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_model_waveforms(self):
        self.sets.model_name = 'waveforms_mlp_ae_v0'
        try:
            self.sets.get_model()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_get_loss_func(self):
        rslt = self.sets.get_loss_func()
        assert type(rslt) == MSELoss

    def test_load_optimizer(self):
        self.sets.model_name = 'waveforms_mlp_ae_v0'
        model = self.sets.get_model()
        rslt = self.sets.load_optimizer(
            model=model,
            learn_rate=0.2
        )
        assert type(rslt) == Adam


class TestCommonPyTorchTrain(TestCase):
    def setUp(self):
        set_data: SettingsDataset = deepcopy(DefaultSettingsDataset)
        set_data.data_type = "MNIST"
        set_train: SettingsPytorch = deepcopy(DefaultSettingsTrainCE)

        self.dut = PyTorchHandler(
            config_train=set_train,
            config_dataset=set_data,
            do_train=True
        )

    def test_saving_path(self):
        rslt = self.dut.get_saving_path()
        self.assertEqual(rslt, "")

    def test_model_number_parameters_non_defined(self):
        try:
            rslt = self.dut.get_number_parameters_from_model
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_model_number_parameters(self):
        model = mnist_mlp_cl_v0()
        self.dut.load_model(
            model=model,
            learn_rate=0.2
        )
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 31910)

    def test_methods_custom_metrics(self):
        try:
            rslt = self.dut.get_epoch_metric_custom_methods
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_define_ptq_level(self):
        self.dut.define_ptq_level(8, 5)

if __name__ == '__main__':
    main()
