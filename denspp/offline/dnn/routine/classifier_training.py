
import numpy as np
from logging import Logger, getLogger
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from datetime import datetime

from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.dnn_handler import SettingsMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainCE, DefaultSettingsTrainMSE
from denspp.offline.dnn import train_classifier_routine, train_autoencoder_routine
from denspp.offline.dnn.plots.plot_dnn import plot_mnist_graphs


def do_train_classifiers(class_dataset, settings: SettingsMLPipeline,
                         yaml_name_index: str='Config_Neural', used_dataset_name: str='quiroga', used_model_name: str='') -> str:
    """Training routine for Classification DL models
    :param class_dataset:           Class of custom-made SettingsDataset from src_dnn/call_dataset.py
    :param settings:                Handler for configuring the routine selection for train deep neural networks
    :param yaml_name_index:         Index of yaml file name
    :param used_dataset_name:       Used dataset name
    :param used_model_name:         Used model for DNN training
    :return:                        String with path to folder in which results are saved
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_type = used_dataset_name
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = used_model_name
    config_train = YamlHandler(
        template=default_train,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_train, default_data

    # --- Loading Data, Build Model and Do Inference
    dataset = prepare_training(
        rawdata=class_dataset(settings=config_data).load_dataset()
    )
    used_model = config_train.get_model(input_size=dataset[0]['in'].size, output_size=dataset.get_cluster_num)
    _, _, path2folder = train_classifier_routine(
        config_ml=settings,
        config_data=config_data,
        config_train=config_train,
        used_dataset=dataset,
        used_model=used_model
    )
    return path2folder


def do_train_autoencoder_classifier(class_dataset, settings: SettingsMLPipeline,
                                    yaml_name_index: str= 'Config_ACL',
                                    model_ae_default_name: str='', model_cl_default_name: str='',
                                    used_dataset_name:str='quiroga') -> dict:
    """Training routine for Autoencoders and Classifier with Encoder after Autoencoder-Training
    Args:
        class_dataset:      Class of custom-made SettingsDataset from src_dnn/call_dataset.py
        settings:           Handler for configuring the routine selection for train deep neural networks
        yaml_name_index:    Index of yaml file name
        model_ae_default_name:  Default name of the autoencoder model
        model_cl_default_name:  Default name of the classifier model
        used_dataset_name:  Default name of the dataset for training [default: quiroga]
    Returns:
        Dictionary with metric results from Autoencoder and Classifier Training
    """
    # --- Processing Step #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_type = used_dataset_name
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Autoencoder Model training
    default_train_ae = deepcopy(DefaultSettingsTrainMSE)
    default_train_ae.model_name = model_ae_default_name
    default_train_ae.custom_metrics = ['dsnr_all']
    config_train_ae = YamlHandler(
        template=default_train_ae,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)

    # --- Loading the YAML file: Classifier Model training
    default_train_cl = deepcopy(DefaultSettingsTrainCE)
    default_train_cl.model_name = model_cl_default_name
    config_train_cl = YamlHandler(
        template=default_train_cl,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_train_cl, default_train_ae, default_data

    # --- Processing Step #1.1: Loading dataset and Build Model
    dataset_ae = get_dataset_ae(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        do_classification=False,
        mode_train_ae=settings.autoencoder_mode,
        noise_std=settings.autoencoder_noise_std
    )
    if settings.autoencoder_feat_size:
        used_model_ae = config_train_ae.get_model(input_size=dataset_ae[0]['in'].size, output_size=settings.autoencoder_feat_size)
    else:
        used_model_ae = config_train_ae.get_model(input_size=dataset_ae[0]['in'].size)

    print("\n# ----------- Step #1: TRAINING AUTOENCODER")
    # --- Processing Step #1.2: Train Autoencoder and Plot Results
    metrics_ae, valid_data_ae, path2folder = train_autoencoder_routine(
        config_ml=settings, config_data=config_data, config_train=config_train_ae,
        used_dataset=dataset_ae, used_model=used_model_ae, path2save=''
    )
    if settings.do_plot:
        used_first_fold = [key for key in metrics_ae.keys()][0]
        results_training(
            path=path2folder, cl_dict=dataset_ae.get_dictionary, feat=valid_data_ae['feat'],
            yin=valid_data_ae['input'], ypred=valid_data_ae['pred'], ymean=dataset_ae.get_mean_waveforms,
            yclus=valid_data_ae['valid_clus'], snr=metrics_ae[used_first_fold]['dsnr_all']
        )
    del dataset_ae

    print("\n# ----------- Step #2: TRAINING CLASSIFIER")
    # --- Processing Step #2.1: Loading dataset and Build Model
    dataset_cl = get_dataset_cl(
        rawdata=class_dataset(settings=config_data).load_dataset(),
        path2model=path2folder
    )
    num_feat = dataset_cl[0]['in'].shape[0] if not settings.autoencoder_feat_size else settings.autoencoder_feat_size
    used_model_cl = config_train_cl.get_model(input_size=num_feat, output_size=dataset_cl.get_cluster_num)

    # --- Processing Step #1.2: Train Classifier
    metrics_cl, valid_data_cl, _ = train_classifier_routine(
        config_ml=settings, config_data=config_data, config_train=config_train_cl,
        used_dataset=dataset_cl, used_model=used_model_cl, path2save=path2folder
    )
    del dataset_cl
    return {
        'path2model_ae': path2folder,
        'metric_ae': metrics_ae,
        'metrics_cl': metrics_cl
    }


def do_train_ae_cl_sweep(class_dataset, settings: SettingsMLPipeline,
                         feat_layer_start: int, feat_layer_inc: int, feat_layer_stop: int,
                         num_epochs_trial: int=50, yaml_name_index: str= 'Config_AECL_Sweep',
                         model_ae_default_name: str='', model_cl_default_name: str='',
                         used_dataset_name:str='quiroga') -> str:
    """Training routine for Autoencoders and Classification after Encoder (Sweep)
    :param class_dataset:           Class of custom-made SettingsDataset from src_dnn/call_dataset.py
    :param settings:                Handler for configuring the routine selection for train deep neural networks
    :param feat_layer_start:        Increasing value for feature layer
    :param feat_layer_inc:          Increasing value for feature layer
    :param feat_layer_stop:         Increasing value for feature layer
    :param num_epochs_trial:        Number of epochs of each run
    :param yaml_name_index:         Index of yaml file name
    :param model_ae_default_name:   Default name for autoencoder model
    :param model_cl_default_name:   Default name for classifier model
    :param used_dataset_name:       Default dataset name used in training
    :return:                        String with path in which the data is saved
    """
    # ------------ STEP #0: Loading YAML files
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_type = used_dataset_name
    config_data = YamlHandler(
        template=default_data,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_Dataset'
    ).get_class(SettingsDataset)

    # --- Loading the YAML file: Autoencoder Model Load and building
    default_ae = deepcopy(DefaultSettingsTrainMSE)
    default_ae.model_name = model_ae_default_name
    default_ae.num_epochs = num_epochs_trial
    config_train_ae = YamlHandler(
        template=default_ae,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainAE'
    ).get_class(ConfigPytorch)

    # --- Loading the YAML file: Classifier Model Load and building
    default_cl = deepcopy(DefaultSettingsTrainCE)
    default_cl.model_name = model_cl_default_name
    default_cl.num_epochs = num_epochs_trial
    config_train_cl = YamlHandler(
        template=default_cl,
        path=settings.get_path2config,
        file_name=f'{yaml_name_index}_TrainCL'
    ).get_class(ConfigPytorch)
    del default_data, default_ae, default_cl

    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_foldername = f'{time_now}_{config_train_ae.model_name}_sweep'
    path2save = config_data.get_path2folder_project / 'runs' / sweep_foldername
    if path2save.exists():
        rmtree(path2save)

    num_clusters = 0
    metrics_runs = dict()
    sweep_val = [idx for idx in range(feat_layer_start, feat_layer_stop, feat_layer_inc)]
    sweep_val.append(feat_layer_stop)
    for idx, feat_size in enumerate(sweep_val):
        path2save_base = f"{path2save}/sweep_{idx:02d}_size{feat_size}"
        # ----------- Step #1: TRAINING AUTOENCODER
        used_dataset_ae = get_dataset_ae(
            rawdata=class_dataset(settings=config_data).load_dataset(),
            mode_train_ae=settings.autoencoder_mode,
            noise_std=settings.autoencoder_noise_std,
            do_classification=False
        )
        used_model_ae = config_train_ae.get_model(input_size=class_dataset[0]['in'].size, output_size=feat_size)
        metrics_ae, valid_data_ae, path2folder = train_autoencoder_routine(
            config_ml=settings,
            config_train=config_train_ae,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_ae,
            used_model=used_model_ae
        )
        del used_dataset_ae, used_model_ae

        # ----------- Step #2: TRAINING CLASSIFIER
        used_dataset_cl = get_dataset_cl(
            rawdata=class_dataset(settings=config_data).load_dataset(),
            path2model=path2folder
        )
        used_model_cl = config_train_cl.get_model(input_size=feat_size, output_size=used_dataset_cl.get_cluster_num)
        metrics_cl = train_classifier_routine(
            config_ml=settings,
            config_train=config_train_cl,
            config_data=config_data,
            path2save=path2save_base,
            used_dataset=used_dataset_cl,
            used_model=used_model_cl
        )
        if idx == 0:
            num_clusters = used_dataset_cl.get_cluster_num

        del used_dataset_cl, used_model_cl
        metrics_runs.update({f"feat_{feat_size:03d}_ae": metrics_ae, f"feat_{feat_size:03d}_cl": metrics_cl})

    metrics_runs.update({'num_clusters': num_clusters})
    # ----------- Step #3: Output results
    np.save(f'{path2save}/results_sweep.npy', metrics_runs, allow_pickle=True)
    return path2save


class PyTorchDatasetHandler:
    _logger: Logger
    _path2save: Path
    _settings: SettingsMLPipeline
    _data: dict

    def __init__(self, settings: SettingsMLPipeline) -> None:
        self._logger = getLogger(__name__)
        self._path2save = Path(".")
        self._settings = settings

    def __plot_mnist_results(self):
        plot_mnist_graphs(
            self._data['input'],
            self._data['valid_clus'],
            "_input",
            path2save=self._path2save
        )
        plot_mnist_graphs(
            self._data['pred'],
            self._data['valid_clus'],
            "_predicted",
            path2save=self._path2save,
            show_plot=self._settings.do_block
        )
