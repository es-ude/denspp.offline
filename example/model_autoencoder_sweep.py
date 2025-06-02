from denspp.offline.template.call_dataset import DatasetLoader

from denspp.offline.yaml_handler import YamlHandler
from denspp.offline.dnn.handler.train_ae_cl_sweep import do_train_ae_cl_sweep
from denspp.offline.dnn.pytorch_config_data import SettingsDataset
from denspp.offline.dnn.dnn_handler import ConfigMLPipeline, DefaultSettings_MLPipe
from denspp.offline.dnn.plots.plot_ae_cl_sweep import extract_data_from_files, plot_common_loss, plot_common_params, \
    plot_architecture_metrics_isolated


settings = SettingsDataset(
    data_path='',
    data_file_name='quiroga',
    use_cell_sort_mode=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=True,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)


if __name__ == "__main__":
    yaml_handler = YamlHandler(DefaultSettings_MLPipe, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(ConfigMLPipeline)
    dnn_handler.do_plot = True
    dnn_handler.do_block = False

    # --- Step #1: Run results
    print("========================================\n Sweep Run for Training Autoencoder + Classification System\n")
    path2save = do_train_ae_cl_sweep(
        rawdata=DatasetLoader(settings=settings).load_dataset(),
        settings=dnn_handler,
        feat_layer_start=1,
        feat_layer_inc=1,
        feat_layer_stop=32
    )

    # --- Step #2: Plot results
    print("===========================================\n Printing results and plot results\n")
    data = extract_data_from_files(path2save)
    plot_common_loss(data, path2save=path2save)
    plot_common_params(data, path2save=path2save)
    plot_architecture_metrics_isolated(data, show_plots=True, path2save=path2save)
    print("\n.done")
