import matplotlib.pyplot as plt
from denspp.offline.dnn.plots.plot_metric import plot_confusion, plot_loss, plot_statistic_data
from denspp.offline.dnn import (
    SettingsMLPipeline,
    SettingsDataset,
    ConfigPytorch,
    TrainClassifier,
    TrainAutoencoder
)


def train_classifier_template(config_ml: SettingsMLPipeline, config_data: SettingsDataset,
                              config_train: ConfigPytorch, used_dataset, used_model,
                              path2save: str='', ptq_quant_lvl: list = (12, 11)) -> tuple[dict, dict, str]:
    """Template for training DL classifiers using PyTorch (incl. plotting)
    Args:
        config_ml:          Settings for handling the ML Pipeline
        config_data:        Settings for handling and loading the dataset (just for saving)
        config_train:       Settings for handling the PyTorch Trainings Routine
        used_dataset:       Used custom-made DataLoader with data set
        used_model:         Used custom-made PyTorch DL model
        path2save:          Path for saving the results [Default: '' --> generate new subfolder in runs
        ptq_quant_lvl:      Quantization level for PTQ [total bitwidth, frac bitwidth]
    Returns:
        Dictionaries with results from training [metrics, validation data] + String to path for saving plots
    """
    # ---Processing Step #1: Preparing Trainings Handler, Build Model
    train_handler = TrainClassifier(config_train=config_train, config_data=config_data, do_train=True)
    train_handler.load_model(model=used_model)
    train_handler.load_data(data_set=used_dataset)
    train_handler.define_ptq_level(ptq_quant_lvl[0], ptq_quant_lvl[1])

    # --- Processing Step #2: Do Training and Validation
    metrics = train_handler.do_training(path2save=path2save, metrics=config_train.custom_metrics)
    path2folder = train_handler.get_saving_path()
    data_result = train_handler.do_validation_after_training()

    # --- Processing Step #3: Plotting
    if config_ml.do_plot:
        plt.close('all')
        used_first_fold = [key for key in metrics.keys()][0]

        plot_loss(metrics[used_first_fold]['acc_train'], metrics[used_first_fold]['acc_valid'],
                  type='Acc.', path2save=path2folder)
        plot_loss(metrics[used_first_fold]['loss_train'], metrics[used_first_fold]['loss_valid'],
                  type=f'{config_train.loss} (CL)', path2save=path2folder)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=path2folder, cl_dict=used_dataset.get_dictionary)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=path2folder, cl_dict=used_dataset.get_dictionary,
                            show_plot=config_ml.do_block)
    # --- Output
    return metrics, data_result, path2folder


def train_autoencoder_template(config_ml: SettingsMLPipeline, config_data: SettingsDataset,
                               config_train: ConfigPytorch, used_dataset, used_model,
                               path2save: str='', ptq_quant_lvl: list = (12, 8)) -> tuple[dict, dict, str]:
    """Template for training DL classifiers using PyTorch (incl. plotting)
    Args:
        config_ml:              Settings for handling the ML Pipeline
        config_data:            Settings for handling and loading the dataset (just for saving)
        config_train:           Settings for handling the PyTorch Trainings Routine
        used_dataset:           Used custom-made DataLoader with data set
        used_model:             Used custom-made PyTorch DL model
        path2save:              Path for saving the results [Default: '' --> generate new subfolder in runs]
        ptq_quant_lvl:          Quantization level for PTQ [total bitwidth, frac bitwidth]
    Returns:
        Dictionaries with results from training [metrics, validation data] + String to path for saving plots
    """
    # ---Processing Step #1: Preparing Trainings Handler, Build Model
    train_handler = TrainAutoencoder(config_train=config_train, config_data=config_data, do_train=True)
    train_handler.load_model(model=used_model)
    train_handler.load_data(data_set=used_dataset)
    train_handler.define_ptq_level(ptq_quant_lvl[0], ptq_quant_lvl[1])

    # --- Processing Step #2: Do Training and Validation
    metrics = train_handler.do_training(path2save=path2save, metrics=config_train.custom_metrics)
    path2folder = train_handler.get_saving_path()
    data_result = train_handler.do_validation_after_training()

    # --- Processing Step #3: Plotting
    if config_ml.do_plot:
        plt.close('all')
        used_first_fold = [key for key in metrics.keys()][0]

        plot_loss(loss_train=metrics[used_first_fold]['loss_train'],
                  loss_valid=metrics[used_first_fold]['loss_valid'],
                  type=config_train.loss, path2save=path2folder)
        plot_statistic_data(train_cl=data_result['train_clus'], valid_cl=data_result['valid_clus'],
                            path2save=path2folder, cl_dict=used_dataset.get_dictionary)
    # --- Output
    return metrics, data_result, path2folder
