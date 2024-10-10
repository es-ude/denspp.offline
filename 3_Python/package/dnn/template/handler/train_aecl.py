import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainMSE, DefaultSettingsTrainCE,
                                           Config_Dataset, DefaultSettingsDataset)
from package.plot.plot_dnn import results_training, plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss

from package.dnn.template.dataset.autoencoder import prepare_training as get_dataset_ae
from package.dnn.template.dataset.autoencoder_class import prepare_training as get_dataset_cl
from package.dnn.pytorch.autoencoder import train_nn as train_autoencoder
from package.dnn.pytorch.classifier import train_nn as train_classifier
import package.dnn.template.models.autoencoder_cnn as models_ae
import package.dnn.template.models.autoencoder_class as models_cl


def do_train_ae_classifier(settings: dnn_handler, num_feat_layer=0, num_output_cl=0,
                           mode_ae=0, noise_std=0.05, add_noise_cluster=False) -> dict:
    """Training routine for Autoencoders and Classifier with Encoder after Autoencoder-Training
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        num_feat_layer:     Size of hidden layer from autoencoder [Default: 0 = take default]
        num_output_cl:      Output size of classifier [Default: 0 = take default]
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
        mode_ae:            Selected model of the Autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input))
        noise_std:          Std of the additional noise added to the input [default: 0.05]
    Returns:
        Dictionary
    """
    metric_run = dict()
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_ACL_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    default_train_ae = DefaultSettingsTrainMSE
    default_train_ae.model_name = models_ae.cnn_ae_v1.__name__
    yaml_nn0 = yaml_config_handler(default_train_ae, yaml_name='Config_ACL_TrainAE')
    config_train_ae = yaml_nn0.get_class(Config_PyTorch)
    del default_train_ae, yaml_nn0

    default_train_cl = DefaultSettingsTrainCE
    default_train_cl.model_name = models_cl.classifier_ae_v1.__name__
    yaml_nn1 = yaml_config_handler(default_train_cl, yaml_name='Config_ACL_TrainCL')
    config_train_cl = yaml_nn1.get_class(Config_PyTorch)
    del default_train_cl, yaml_nn1

    print("\n# ----------- Step #1: TRAINING AUTOENCODER")
    # --- Processing: Loading dataset and Do Autoencoder Training
    dataset = get_dataset_ae(settings=config_data, do_classification=False,
                             mode_train_ae=mode_ae, noise_std=noise_std, add_noise_cluster=add_noise_cluster)

    train_handler = train_autoencoder(config_train=config_train_ae, config_data=config_data)
    if num_feat_layer:
        used_model_ae = models_ae.models_available.build_model(config_train_ae.model_name)
    else:
        used_model_ae = models_ae.models_available.build_model(config_train_ae.model_name, output_size=num_feat_layer)
    train_handler.load_model(used_model_ae)
    train_handler.load_data(dataset)
    metrics_ae = train_handler.do_training(metrics=['snr'])
    path2save = train_handler.get_saving_path()

    if settings.do_plot:
        plt.close('all')
        used_first_fold = [key for key in metrics_ae.keys()][0]
        data_result = train_handler.do_validation_after_training()
        data_mean = dataset.frames_me

        plot_loss(loss_train=metrics_ae[used_first_fold]['loss_train'],
                  loss_valid=metrics_ae[used_first_fold]['loss_valid'],
                  type=config_train_ae.loss, path2save=path2save)
        results_training(
            path=path2save, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
            yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
            yclus=data_result['valid_clus'], snr=metrics_ae[used_first_fold]['snr']
        )
    del dataset, train_handler

    print("\n# ----------- Step #2: TRAINING CLASSIFIER")
    # --- Processing: Loading dataset and Do Classification
    dataset = get_dataset_cl(settings=config_data, path2model=path2save, add_noise_cluster=add_noise_cluster)

    train_handler = train_classifier(config_train=config_train_cl, config_data=config_data)
    num_feat = dataset[0]['in'].shape[0] if not num_feat_layer else num_feat_layer
    used_model_cl = models_cl.models_available.build_model(config_train_cl.model_name, input_size=num_feat)
    train_handler.load_model(used_model_cl)
    train_handler.load_data(dataset)
    metrics_cl = train_handler.do_training(path2save)

    if settings.do_plot:
        data_result = train_handler.do_validation_after_training()
        used_first_fold = [key for key in metrics_ae.keys()][0]

        plot_loss(metrics_cl[used_first_fold]['train_acc'], metrics_cl[used_first_fold]['valid_acc'],
                  type='Acc. (CL)', path2save=path2save)
        plot_loss(metrics_cl[used_first_fold]['train_loss'], metrics_cl[used_first_fold]['valid_loss'],
                  type=f'{config_train_cl.loss} (CL)', path2save=path2save)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       cl_dict=data_result['cl_dict'], path2save=path2save,
                       name_addon="training")
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=path2save, cl_dict=data_result['cl_dict'],
                            show_plot=settings.do_block)
    del dataset, train_handler

    # --- Generierung output
    metric_run.update({"path2model_ae": path2save,
                       'metric_ae': metrics_ae, 'metrics_cl': metrics_cl})
    return metric_run


if __name__ == "__main__":
    sets = dnn_handler(
        mode_train_dnn=0,
        do_plot=True,
        do_block=True
    )
    result = do_train_ae_classifier(sets)
    print(result)
