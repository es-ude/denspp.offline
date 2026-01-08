from unittest import TestCase, main
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from denspp.offline import get_path_to_project
from denspp.offline.dnn import (
    SettingsDataset,
    DefaultSettingsDataset,
    DatasetFromFile
)
from .classifier_train import TrainClassifier, SettingsClassifier, DefaultSettingsTrainingCE
from .dataset_dummy import generate_dummy_dataset, dummy_mlp_cl_v0


class TestPyTorchModelConfigClassifier(TestCase):
    def setUp(self):
        self.sets: SettingsClassifier = deepcopy(DefaultSettingsTrainingCE)

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


class TestClassifierTraining(TestCase):
    def setUp(self):
        self.set_train: SettingsClassifier = deepcopy(DefaultSettingsTrainingCE)
        self.set_train.num_epochs = 10
        self.set_train.model_name = dummy_mlp_cl_v0.__name__
        self.set_dataset: SettingsDataset = deepcopy(DefaultSettingsDataset)
        self.set_dataset.data_type = 'dummy'
        self.dataset: DatasetFromFile = generate_dummy_dataset(2048, 100)

        self.dut = TrainClassifier(
            config_train=self.set_train,
            config_data=self.set_dataset,
            do_train=True
        )

    def test_saving_path(self):
        rslt = self.dut.get_saving_path()
        self.assertEqual(str(rslt), get_path_to_project())

    def test_number_parameters(self):
        self.dut.load_model(
            model=dummy_mlp_cl_v0(input_size=self.dataset.data.shape[1]),
            learn_rate=0.1
        )
        rslt = self.dut.get_number_parameters_from_model
        self.assertEqual(rslt, 4206)

    def test_define_ptq(self):
        self.dut.define_ptq_level(8, 6)

    def test_custom_metric_methods(self):
        rslt = self.dut.get_epoch_metric_custom_methods
        self.assertEqual(rslt, ['accuracy', 'precision', 'recall', 'fbeta', 'ptq_acc'])

    def test_training_phase(self):
        self.dut.load_dataset(
            dataset=self.dataset,
        )
        self.dut.load_model(
            model=dummy_mlp_cl_v0(input_size=self.dataset.data.shape[1]),
            learn_rate=0.1
        )
        metric = self.dut.do_training()
        self.assertEqual(len(metric), 1)
        self.assertEqual(list(metric['fold_000'].keys()), ['acc_train', 'acc_valid', 'loss_train', 'loss_valid'])
        self.assertEqual(len(metric['fold_000']['loss_train']), 10)
        self.assertEqual(len(metric['fold_000']['loss_valid']), 10)
        self.assertEqual(len(metric['fold_000']['acc_train']), 10)
        self.assertEqual(len(metric['fold_000']['acc_train']), 10)

        overview = self.dut.get_best_model('cl')
        self.assertGreater(len(overview), 0)

    def test_post_validation_without_training(self):
        try:
            self.dut.do_post_training_validation()
        except RuntimeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_post_validation_without_ptq(self):
        self.dut.load_dataset(
            dataset=self.dataset
        )
        self.dut.load_model(
            model=dummy_mlp_cl_v0(input_size=self.dataset.data.shape[1]),
            learn_rate=0.1
        )
        self.dut.do_training()
        rslt = self.dut.do_post_training_validation(do_ptq=False)
        self.assertEqual(rslt.label_names, self.dataset.dict)

    def test_post_validation_with_ptq(self):
        self.dut.load_dataset(
            dataset=self.dataset,
        )
        self.dut.load_model(
            model=dummy_mlp_cl_v0(input_size=self.dataset.data.shape[1]),
            learn_rate=0.1
        )
        self.dut.do_training()
        rslt = self.dut.do_post_training_validation(do_ptq=True)
        self.assertEqual(rslt.label_names, self.dataset.dict)


if __name__ == '__main__':
    main()
