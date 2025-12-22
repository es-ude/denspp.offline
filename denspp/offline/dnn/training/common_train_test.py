from copy import deepcopy
from unittest import TestCase, main

from denspp.offline import get_path_to_project
from denspp.offline.dnn.data_config import (
    SettingsDataset,
    DefaultSettingsDataset
)
from denspp.offline.dnn.models.mnist import mnist_mlp_cl_v0, mnist_mlp_ae_v0
from .common_train import (
    SettingsPytorch,
    PyTorchHandler
)


# --- Info: Function have to start with test_*
class TestCommonPyTorchTrain(TestCase):
    def setUp(self):
        set_data: SettingsDataset = deepcopy(DefaultSettingsDataset)
        set_data.data_type = "MNIST"
        set_train = SettingsPytorch(
            model_name='',
            patience=20,
            optimizer='Adam',
            loss='Cross Entropy',
            num_kfold=1,
            num_epochs=10,
            batch_size=256,
            data_do_shuffle=True,
            data_split_ratio=0.2,
            deterministic_do=False,
            deterministic_seed=42,
            custom_metrics=[]
        )
        self.dut = PyTorchHandler(
            config_train=set_train,
            config_dataset=set_data,
            do_train=True
        )

    def test_saving_path(self):
        rslt = self.dut.get_saving_path()
        self.assertEqual(str(rslt), get_path_to_project())

    def test_get_model(self):
        self.dut._settings_train.model_name = mnist_mlp_cl_v0.__name__
        rslt = self.dut._settings_train.get_model()
        self.assertEqual(type(rslt), mnist_mlp_cl_v0)

    def test_get_signature(self):
        self.dut._settings_train.model_name = mnist_mlp_cl_v0.__name__
        rslt = self.dut._settings_train.get_signature()
        self.assertEqual(rslt, ['input_size', 'output_size'])

    def test_model_number_parameters_non_defined(self):
        try:
            rslt = self.dut.get_number_parameters_from_model
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_model_number_parameters_cl(self):
        model = mnist_mlp_cl_v0()
        self.dut.load_model(
            model=model,
            learn_rate=0.2
        )
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 31910)

    def test_model_number_parameters_ae(self):
        model = mnist_mlp_ae_v0()
        self.dut.load_model(
            model=model,
            learn_rate=0.2
        )
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 64574)

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
