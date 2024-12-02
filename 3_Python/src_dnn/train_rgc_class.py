from copy import deepcopy
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier
from package.data_process.rgc_combination import rgc_logic_combination

from package.dnn.template.dataset.classifier import prepare_training
import src_dnn.models.rgc_onoff_class as models


def do_train_rgc_class(settings: Config_ML_Pipeline, yaml_name_index='Config_RGC') -> None:
    """Training routine for classifying RGC ON/OFF and Transient/Sustained Types (Classification)
    Args:
        settings:           Handler for configuring the routine selection to train deep neural networks
        yaml_name_index:    Index of yaml file name
    """
    # --- Loading the YAML file: Dataset
    default_data = deepcopy(DefaultSettingsDataset)
    default_data.data_file_name = '2023-11-24_Dataset-07_RGC_TDB_Merged.npy'
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = deepcopy(DefaultSettingsTrainCE)
    default_train.model_name = models.dnn_rgc_v1.__name__
    yaml_train = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)

    # ---Loading Data, Build Model and Do Training
    used_dataset = prepare_training(config_data)
    used_model = models.models_available.build_model(config_train.model_name)
    _, _, path2folder = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=used_dataset, used_model=used_model, path2save=''
    )

    # --- Plotting reduced model (ON/OFF and Transient/Sustained)
    if settings.do_plot:
        rgc_logic_combination(path2folder, show_plot=settings.do_block)


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_rgc_class(DefaultSettings_MLPipe)
