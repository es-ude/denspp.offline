import unittest
from copy import deepcopy
from denspp.offline.template.call_data_dummy import DataLoaderTest
from denspp.offline.dnn import (
    ConfigPytorch, DefaultSettingsTrainCE,
    SettingsDataset, DefaultSettingsDataset,
    DefaultSettingsMLPipeline
)
from denspp.offline.dnn.models.autoencoder_class import synthetic_ae_cl_v1
from .classifier import TrainClassifier, train_classifier_routine


class TestClassifierTraining(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skip
    def test_train_classifier(self):
        set_pytorch: ConfigPytorch = deepcopy(DefaultSettingsTrainCE)
        set_pytorch.model_name = 'synthetic_ae_cl_v1'
        set_pytorch.num_epochs = 1
        set_dataset: SettingsDataset = deepcopy(DefaultSettingsDataset)
        set_dataset.data_file_name = ''

        rslt = train_classifier_routine(
            config_ml=DefaultSettingsMLPipeline,
            config_data=DefaultSettingsDataset,
            config_train=set_pytorch,
            used_dataset=DataLoaderTest,
            used_model=synthetic_ae_cl_v1,
            path2save='',
            ptq_quant_lvl=[8,7]
        )
        self.assertEqual(len(rslt), 3)


if __name__ == '__main__':
    unittest.main()
