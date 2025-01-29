from package.dnn.dnn_handler import ConfigMLPipeline
from package.dnn.handler.train_cl import do_train_spike_class
from package.dnn.handler.train_ae_cl import do_train_ae_classifier
from package.data_process.rgc_combination import rgc_logic_combination


def do_train_rgc_class(settings: ConfigMLPipeline) -> None:
    """Training routine for classifying RGC ON/OFF and Transient/Sustained Types (Classification)
    Args:
        settings:           Handler for configuring the routine selection to train deep neural networks
    Returns:
        String with path to folder in which results are saved
    """
    path2folder = do_train_spike_class(
        settings=settings,
        yaml_name_index='Config_RGC',
        used_dataset_name='rgc_tdb',
        used_model_name='rgc_onoff_tdb_dnn_cl_v1'
    )
    if settings.do_plot:
        rgc_logic_combination(path2folder, show_plot=settings.do_block)


def do_train_rgc_aecl(settings: ConfigMLPipeline, add_noise_cluster: bool=False) -> None:
    """Training routine for Autoencoders and Classification after Encoder for Retinal Ganglion Celltype Classification
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
        add_noise_cluster:  Adding noise cluster to dataset [Default: False]
    Returns:
        Dictionary with metric results from Autoencoder and Classifier Training
    """
    results = do_train_ae_classifier(
        settings=settings,
        yaml_name_index='Config_RGC',
        model_ae_default_name='',
        model_cl_default_name='',
        used_dataset_name='rgc_tdb',
        add_noise_cluster=add_noise_cluster
    )
    if settings.do_plot:
        rgc_logic_combination(results['path2model_ae'], show_plot=settings.do_block)
