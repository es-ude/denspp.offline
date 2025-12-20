import unittest
from copy import deepcopy
from denspp.offline.dnn.models import mnist, waveforms
from denspp.offline.dnn import (
    ConfigPytorch, DefaultSettingsTrainMSE,
    SettingsDataset, DefaultSettingsDataset,
    SettingsMLPipeline, DefaultSettingsMLPipeline,
)
from denspp.offline.dnn.routine import DatasetAutoencoder
from denspp.offline.template.call_dataset import DatasetLoader
from .autoencoder import TrainAutoencoder, train_autoencoder_routine


class TestAutoencoderTraining(unittest.TestCase):
    def setUp(self):
        self.set_routine: SettingsMLPipeline = deepcopy(DefaultSettingsMLPipeline)
        self.set_routine.do_plot = False
        self.set_train_ae: ConfigPytorch = deepcopy(DefaultSettingsTrainMSE)
        self.set_train_ae.num_epochs = 1
        self.set_dataset: SettingsDataset = deepcopy(DefaultSettingsDataset)

    def test_train_mnist_normal(self):
        self.set_dataset.data_type = 'MNIST'
        self.set_train_ae.model_name = str(mnist.mnist_mlp_ae_v0)
        dataset = DatasetAutoencoder(
            dataset=DatasetLoader(self.set_dataset).load_dataset(),
            noise_std=0.1,
            mode_train=0
        )
        rslt = train_autoencoder_routine(
            config_ml=self.set_routine,
            config_data=self.set_dataset,
            config_train=self.set_train_ae,
            used_dataset=dataset,
            used_model=mnist.mnist_mlp_ae_v0(),
            path2save='',
            ptq_quant_lvl=[8, 7]
        )
        self.assertEqual(len(rslt), 3)

    def test_train_waveforms_normal(self):
        self.set_dataset.data_type = 'WAVEFORMS'
        self.set_train_ae.model_name = str(waveforms.waveforms_mlp_ae_v0)
        dataset = DatasetAutoencoder(
            dataset=DatasetLoader(self.set_dataset).load_dataset(),
            noise_std=0.1,
            mode_train=0
        )

        rslt = train_autoencoder_routine(
            config_ml=self.set_routine,
            config_data=self.set_dataset,
            config_train=self.set_train_ae,
            used_dataset=dataset,
            used_model=waveforms.waveforms_mlp_ae_v0(input_size=dataset[0]['in'].shape[0]),
            path2save='',
            ptq_quant_lvl=[8, 7]
        )
        self.assertEqual(len(rslt), 3)

    def test_train_waveforms_denoising_mean(self):
        self.set_dataset.data_type = 'WAVEFORMS'
        self.set_train_ae.model_name = str(waveforms.waveforms_mlp_ae_v0)
        dataset = DatasetAutoencoder(
            dataset=DatasetLoader(self.set_dataset).load_dataset(),
            noise_std=0.1,
            mode_train=1
        )
        rslt = train_autoencoder_routine(
            config_ml=self.set_routine,
            config_data=self.set_dataset,
            config_train=self.set_train_ae,
            used_dataset=dataset,
            used_model=waveforms.waveforms_mlp_ae_v0(input_size=dataset[0]['in'].shape[0]),
            path2save='',
            ptq_quant_lvl=[8, 7]
        )
        self.assertEqual(len(rslt), 3)

    def test_train_waveforms_denoising_random(self):
        self.set_dataset.data_type = 'WAVEFORMS'
        self.set_train_ae.model_name = str(waveforms.waveforms_mlp_ae_v0)
        dataset = DatasetAutoencoder(
            dataset=DatasetLoader(self.set_dataset).load_dataset(),
            noise_std=0.1,
            mode_train=2
        )
        rslt = train_autoencoder_routine(
            config_ml=self.set_routine,
            config_data=self.set_dataset,
            config_train=self.set_train_ae,
            used_dataset=dataset,
            used_model=waveforms.waveforms_mlp_ae_v0(input_size=dataset[0]['in'].shape[0]),
            path2save='',
            ptq_quant_lvl=[8, 7]
        )
        self.assertEqual(len(rslt), 3)

    def test_train_waveforms_denoising_gaussian(self):
        self.set_dataset.data_type = 'WAVEFORMS'
        self.set_train_ae.model_name = str(waveforms.waveforms_mlp_ae_v0)
        dataset = DatasetAutoencoder(
            dataset=DatasetLoader(self.set_dataset).load_dataset(),
            noise_std=0.1,
            mode_train=3
        )
        rslt = train_autoencoder_routine(
            config_ml=self.set_routine,
            config_data=self.set_dataset,
            config_train=self.set_train_ae,
            used_dataset=dataset,
            used_model=waveforms.waveforms_mlp_ae_v0(input_size=dataset[0]['in'].shape[0]),
            path2save='',
            ptq_quant_lvl=[8, 7]
        )
        self.assertEqual(len(rslt), 3)


if __name__ == '__main__':
    unittest.main()
