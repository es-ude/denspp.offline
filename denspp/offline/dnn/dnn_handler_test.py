import numpy as np
from unittest import TestCase, main, skip
from pathlib import Path
from shutil import rmtree

from denspp.offline import get_path_to_project, check_keylist_elements_all
from denspp.offline.dnn import DatasetFromFile
from denspp.offline.dnn.models.mnist import mnist_mlp_cl_v0, mnist_mlp_ae_v0
from denspp.offline.dnn.models.waveforms import waveforms_mlp_cl_v0, waveforms_mlp_ae_v0
from denspp.offline.dnn.dnn_handler import TrainingResults, PyTorchTrainer, DataValidation, SettingsTraining


class TestPytorchTrainer(TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(get_path_to_project('temp_test'), ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        rmtree(get_path_to_project('temp_test'), ignore_errors=True)
        rmtree(get_path_to_project('runs'), ignore_errors=True)
        Path(get_path_to_project('runs')).mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.sets_ae = SettingsTraining(
            mode_train=1,
            do_block=False,
            do_ptq=False,
            ptq_total_bitwidth=8,
            ptq_frac_bitwidth=6,
        )
        self.sets_cl = SettingsTraining(
            mode_train=0,
            do_block=False,
            do_ptq=False,
            ptq_total_bitwidth=8,
            ptq_frac_bitwidth=6,
        )
        self.dut = PyTorchTrainer(
            use_case='MNIST',
            path2config='temp_test',
            default_model=mnist_mlp_cl_v0.__name__
        )

    def test_path2config(self):
        rslt = self.dut.path2config
        self.assertEqual(str(rslt), get_path_to_project('temp_test'))

    def test_check_config_available(self):
        self.assertTrue((self.dut.path2config / "ConfigClassifier_MNIST.json").exists())
        self.assertTrue((self.dut.path2config / "ConfigDataset_MNIST.json").exists())
        self.assertTrue((self.dut.path2config / "ConfigTraining_MNIST.json").exists())

        self.assertTrue(self.dut.config_available)
        rmtree(self.dut.path2config)
        self.assertFalse(self.dut.config_available)

    def test_get_custom_metrics_classifier(self):
        rslt = PyTorchTrainer(
            use_case='MNIST',
            settings=self.sets_cl,
            path2config='temp_test/method_cl',
            default_model=mnist_mlp_cl_v0.__name__,
            generate_configs=False
        ).get_custom_metric_calculation
        self.assertEqual(rslt, ['accuracy', 'precision', 'recall', 'fbeta', 'ptq_acc'])

    def test_get_custom_metrics_autoencoder(self):
        rslt = PyTorchTrainer(
            use_case='MNIST',
            settings=self.sets_ae,
            path2config='temp_test/method_ae',
            default_model=mnist_mlp_ae_v0.__name__,
            generate_configs=False
        ).get_custom_metric_calculation
        self.assertEqual(rslt, ['snr_in', 'snr_out', 'dsnr_all', 'ptq_loss'])

    def test_start_training_first_with_init(self):
        rmtree(self.dut.path2config)
        dut = PyTorchTrainer(
            use_case='MNIST',
            path2config='temp_test',
            default_model=mnist_mlp_cl_v0.__name__
        )
        try:
            dut.do_training()
        except AttributeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_get_model_overview(self):
        overview = self.dut._settings_model.get_model_overview(print_overview=True)
        self.assertGreater(len(overview), 2)

    def test_get_model(self):
        model = self.dut.get_model()
        self.assertEqual(type(model), mnist_mlp_cl_v0)

    def test_get_dataset_mnist(self):
        dut = PyTorchTrainer(
            use_case='MNIST',
            settings=self.sets_cl,
            path2config='temp_test',
            default_model=mnist_mlp_cl_v0.__name__,
        )
        rslt = dut.get_dataset()
        self.assertEqual(type(rslt), DatasetFromFile)
        self.assertEqual(rslt.data.shape, (70000, 28, 28))
        self.assertEqual(len(rslt.dict), 10)
        self.assertEqual(rslt.dict[0], 'zero')
        self.assertEqual(rslt.label.shape, (70000, ))
        self.assertEqual(rslt.mean.shape, (10, 28, 28))
        dut.do_plot_dataset(Path(get_path_to_project('temp_test')))

    def test_get_dataset_waveforms(self):
        dut = PyTorchTrainer(
            use_case='Waveforms',
            settings=self.sets_cl,
            path2config='temp_test',
            default_model=waveforms_mlp_cl_v0.__name__,
        )
        rslt = dut.get_dataset()
        self.assertEqual(type(rslt), DatasetFromFile)
        self.assertEqual(rslt.data.shape, (12000, 280))
        self.assertEqual(len(rslt.dict), 12)
        self.assertEqual(rslt.dict[0], 'RECT_HALF')
        self.assertEqual(rslt.label.shape, (12000,))
        self.assertEqual(rslt.mean.shape, (12, 280))
        dut.do_plot_dataset(Path(get_path_to_project('temp_test')))

    def test_save_load_results(self):
        path2save = Path(get_path_to_project("temp_test"))
        rslt = TrainingResults(
            metrics={},
            settings={},
            data=DataValidation(
                input=np.random.randn(100, 28, 28),
                valid_label=np.random.randint(100),
                train_label=np.random.randint(100),
                feat=None,
                mean=None,
                output=np.random.randint(100, ),
                label_names=[]
            ),
            path=Path("."),
            metrics_custom=['Accuracy'],
        )
        # --- Part #1: Saving file
        self.dut._save_results(
            data=rslt,
            path2save=path2save,
            addon='test'
        )
        self.assertTrue((path2save / 'results_test.npy').exists())
        # --- Part #2: Loading file and check content
        data = self.dut._load_results(path2save / 'results_test.npy')
        self.assertEqual(type(rslt.data), type(data.data))
        self.assertEqual(rslt.metrics, data.metrics)
        self.assertEqual(rslt.metrics_custom, data.metrics_custom)
        self.assertEqual(rslt.path, data.path)
        self.assertEqual(rslt.settings, data.settings)

    def test_train_classifier_mnist(self):
        sets = dict(
            use_case='MNIST',
            settings=self.sets_cl,
            path2config='temp_test/mnist_cl',
            default_model=mnist_mlp_cl_v0.__name__,
        )
        PyTorchTrainer(**sets)
        dut = PyTorchTrainer(**sets)
        try:
            rslt = dut.do_training()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(check_keylist_elements_all(list(rslt.metrics['fold_000'].keys()), ['loss_train', 'loss_valid']))
            self.assertTrue(type(rslt.data), DataValidation)
            self.assertTrue(get_path_to_project() in str(rslt.path))
            dut.do_plot_results(rslt)

    def test_train_classifier_waveforms(self):
        sets = dict(
            use_case='WAVEFORMS',
            settings=self.sets_cl,
            path2config='temp_test/wave_cl',
            default_model=waveforms_mlp_cl_v0.__name__,
        )
        PyTorchTrainer(**sets)
        dut = PyTorchTrainer(**sets)
        try:
            rslt = dut.do_training()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(check_keylist_elements_all(list(rslt.metrics['fold_000'].keys()), ['loss_train', 'loss_valid']))
            self.assertTrue(type(rslt.data), DataValidation)
            self.assertTrue(get_path_to_project() in str(rslt.path))
            dut.do_plot_results(rslt)

    def test_train_classifier_plotting(self):
        hndl = PyTorchTrainer(
            use_case='Waveforms',
            settings=self.sets_cl,
            path2config='temp_test',
            generate_configs=False
        )
        path2file = Path(get_path_to_project("runs"))
        path2runs = list(path2file.glob('*_cl_*'))[-1]
        rslt = hndl.read_file_and_plot(
            path2file=path2runs / 'results_cl.npy',
            epoch_zoom=None,
            do_plot=False
        )
        self.assertEqual(type(rslt), TrainingResults)

    @skip("Skipping this test")
    def test_train_classifier_plotting_zoomed(self):
        hndl = PyTorchTrainer(
            use_case='Waveforms',
            settings=self.sets_cl,
            path2config='temp_test',
            generate_configs=False
        )
        path2file = Path(get_path_to_project("runs"))
        path2runs = list(path2file.glob('*_cl_*'))[-1]
        rslt = hndl.read_file_and_plot(
            path2file=path2runs / 'results_cl.npy',
            epoch_zoom=[8, 10],
            do_plot=False
        )
        self.assertEqual(type(rslt), TrainingResults)

    def test_train_autoencoder_mnist(self):
        sets = dict(
            use_case='MNIST',
            settings=self.sets_ae,
            path2config='temp_test/mnist_ae',
            default_model=mnist_mlp_ae_v0.__name__,
        )
        PyTorchTrainer(**sets)
        dut = PyTorchTrainer(**sets)
        try:
            rslt = dut.do_training()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(check_keylist_elements_all(list(rslt.metrics['fold_000'].keys()), ['loss_train', 'loss_valid']))
            self.assertTrue(type(rslt.data), DataValidation)
            self.assertTrue(get_path_to_project() in str(rslt.path))
            dut.do_plot_results(rslt)

    def test_train_autoencoder_waveforms(self):
        sets = dict(
            use_case='WAVEFORMS',
            settings=self.sets_ae,
            path2config='temp_test/wave_ae',
            default_model=waveforms_mlp_ae_v0.__name__,
        )
        PyTorchTrainer(**sets)
        dut = PyTorchTrainer(**sets)
        try:
            rslt = dut.do_training()
        except AttributeError:
            self.assertTrue(False)
        else:
            self.assertTrue(check_keylist_elements_all(list(rslt.metrics['fold_000'].keys()), ['loss_train', 'loss_valid']))
            self.assertTrue(type(rslt.data), DataValidation)
            self.assertTrue(get_path_to_project() in str(rslt.path))
            dut.do_plot_results(rslt)

    def test_train_autoencoder_plotting(self):
        hndl = PyTorchTrainer(
            use_case='Waveforms',
            settings=self.sets_ae,
            path2config='temp_test',
            generate_configs=False
        )
        path2file = Path(get_path_to_project("runs"))
        path2runs = list(path2file.glob('*_ae_*'))[-1]
        rslt = hndl.read_file_and_plot(
            path2file=path2runs / 'results_ae.npy',
            epoch_zoom=None,
            do_plot=False
        )
        self.assertEqual(type(rslt), TrainingResults)

    @skip("Skipping this test")
    def test_train_autoencoder_plotting_zoomed(self):
        hndl = PyTorchTrainer(
            use_case='Waveforms',
            settings=self.sets_ae,
            path2config='temp_test',
            generate_configs=False
        )
        path2file = Path(get_path_to_project("runs"))
        path2runs = list(path2file.glob('*_ae_*'))[-1]
        rslt = hndl.read_file_and_plot(
            path2file=path2runs / 'results_ae.npy',
            epoch_zoom=[8, 10],
            do_plot=False
        )
        self.assertEqual(type(rslt), TrainingResults)


if __name__ == '__main__':
    main()
