import matplotlib.pyplot as plt
from package.plot.plot_dnn import plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_config_data import Config_Dataset
from package.dnn.pytorch_config_model import Config_PyTorch
from package.dnn.pytorch.classifier import train_nn as train_nn_cl
from package.dnn.pytorch.autoencoder import train_nn as train_nn_ae


def do_train_classifier(config_ml: Config_ML_Pipeline, config_data: Config_Dataset,
                        config_train: Config_PyTorch, used_dataset, used_model,
                        path2save='', calc_custom_metrics=(), print_results=True) -> [dict, dict, str]:
    """Template for training DL classifiers using PyTorch (incl. plotting)
    Args:
        config_ml:          Settings for handling the ML Pipeline
        config_data:        Settings for handling and loading the dataset (just for saving)
        config_train:       Settings for handling the PyTorch Trainings Routine
        used_dataset:       Used custom-made DataLoader with data set
        used_model:         Used custom-made PyTorch DL model
        path2save:          Path for saving the results [Default: '' --> generate new subfolder in runs]
        calc_custom_metrics:List with metric names (custom-made) to determine during trainings process
        print_results:      Printing the results into Terminal
    Returns:
        Dictionaries with results from training [metrics, validation data] + String to path for saving plots
    """
    # ---Processing Step #1: Preparing Trainings Handler, Build Model
    train_handler = train_nn_cl(config_train=config_train, config_data=config_data, do_train=True)
    train_handler.load_model(model=used_model, print_model=print_results)
    train_handler.load_data(data_set=used_dataset)
    train_handler.get_metric_methods()

    # --- Processing Step #2: Do Training and Validation
    metrics = train_handler.do_training(path2save=path2save, metrics=calc_custom_metrics)
    path2folder = train_handler.get_saving_path()
    data_result = train_handler.do_validation_after_training()

    # --- Processing Step #3: Plotting
    if config_ml.do_plot:
        plt.close('all')
        used_first_fold = [key for key in metrics.keys()][0]

        plot_loss(metrics[used_first_fold]['train_acc'], metrics[used_first_fold]['valid_acc'],
                  type='Acc.', path2save=path2folder)
        plot_loss(metrics[used_first_fold]['train_loss'], metrics[used_first_fold]['valid_loss'],
                  type=f'{config_train.loss} (CL)', path2save=path2folder)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=path2folder, cl_dict=used_dataset.get_dictionary)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=path2folder, cl_dict=used_dataset.get_dictionary,
                            show_plot=config_ml.do_block)
    # --- Output
    return metrics, data_result, path2folder


def do_train_autoencoder(config_ml: Config_ML_Pipeline, config_data: Config_Dataset,
                         config_train: Config_PyTorch, used_dataset, used_model,
                         path2save='', calc_custom_metrics=(), save_vhdl=False, path4vhdl='', print_results=True) -> [dict, dict, str]:
    """Template for training DL classifiers using PyTorch (incl. plotting)
    Args:
        config_ml:              Settings for handling the ML Pipeline
        config_data:            Settings for handling and loading the dataset (just for saving)
        config_train:           Settings for handling the PyTorch Trainings Routine
        used_dataset:           Used custom-made DataLoader with data set
        used_model:             Used custom-made PyTorch DL model
        path2save:              Path for saving the results [Default: '' --> generate new subfolder in runs]
        calc_custom_metrics:    List with metric names (custom-made) to determine during trainings process
        print_results:  Printing the results into Terminal
    Returns:
        Dictionaries with results from training [metrics, validation data] + String to path for saving plots
    """
    # ---Processing Step #1: Preparing Trainings Handler, Build Model
    train_handler = train_nn_ae(config_train=config_train, config_data=config_data, do_train=True)
    train_handler.load_model(model=used_model, print_model=print_results)
    train_handler.load_data(data_set=used_dataset)

    # --- Processing Step #2: Do Training and Validation
    train_handler.get_metric_methods()
    metrics = train_handler.do_training(path2save=path2save, metrics=calc_custom_metrics)
    path2folder = train_handler.get_saving_path()
    data_result = train_handler.do_validation_after_training()

    # --- Save VHDL Code
    if save_vhdl:
        train_handler.save_model_to_vhdl(path4vhdl=path4vhdl)

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
