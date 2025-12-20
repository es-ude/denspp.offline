import unittest
from copy import deepcopy

from denspp.offline.dnn import (
    SettingsPytorch,
    DefaultSettingsTrainCE,
    SettingsDataset,
    DefaultSettingsDataset,
    SettingsMLPipeline,
    DefaultSettingsMLPipeline,
    DatasetFromFile
)
from .classifier_train import TrainClassifier
from .dataset_dummy import generate_dummy_dataset, dummy_mlp_cl_v0


class TestClassifierTraining(unittest.TestCase):
    def setUp(self):
        self.set_routine: SettingsMLPipeline = deepcopy(DefaultSettingsMLPipeline)
        self.set_routine.do_plot = False
        self.set_train: SettingsPytorch = deepcopy(DefaultSettingsTrainCE)
        self.set_train.num_epochs = 10
        self.set_train.model_name = str(dummy_mlp_cl_v0)
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
        self.assertEqual(rslt, "")

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
        self.assertEqual(rslt, ['accuracy', 'precision', 'recall', 'fbeta', 'ptq_loss'])

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
        self.assertEqual(rslt['settings'], self.set_train)
        self.assertEqual(rslt['cl_dict'], self.dataset.dict)

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
        self.assertEqual(rslt['settings'], self.set_train)
        self.assertEqual(rslt['cl_dict'], self.dataset.dict)


if __name__ == '__main__':
    unittest.main()
