from denspp.offline.dnn.dnn_handler import ConfigMLPipeline, DefaultSettings_MLPipe
from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.structure_builder import init_project_folder
from src_dnn.call_dataset import DatasetLoader


if __name__ == "__main__":
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)"
          "\n===========================================================================================")

    # --- Loading YAML-Settings file
    init_project_folder()
    yaml_handler = YamlConfigHandler(DefaultSettings_MLPipe, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(ConfigMLPipeline)

    # --- Selecting model for train
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_cl
            do_train_torchvision_cl(DatasetLoader, dnn_handler, 'mnist')
        case 1:
            # --- MNIST (Autoencoder)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_ae
            do_train_torchvision_ae(DatasetLoader, dnn_handler, 'mnist')
        case 2:
            # --- Fashion-MNIST (Classifier)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_cl
            do_train_torchvision_cl(DatasetLoader, dnn_handler, 'fashion')
        case 3:
            # --- FashionMNIST (Autoencoder)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_ae
            do_train_torchvision_ae(DatasetLoader, dnn_handler, 'fashion')
        case 4:
            # --- CIFAR-10 (Classifier)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_cl
            do_train_torchvision_cl(DatasetLoader, dnn_handler, 'cifar-10')
        case 5:
            # --- CIFAR-10 (Autoencoder)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_ae
            do_train_torchvision_ae(DatasetLoader, dnn_handler, 'cifar-10')
        case 6:
            # --- CIFAR-100 (Classifier)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_cl
            do_train_torchvision_cl(DatasetLoader, dnn_handler, 'cifar-100')
        case 7:
            # --- CIFAR-100 (Autoencoder)
            from denspp.offline.dnn.handler.train_torchvision import do_train_torchvision_ae
            do_train_torchvision_ae(DatasetLoader, dnn_handler, 'cifar-100')
        case 8:
            # --- Waveform Classification
            from denspp.offline.dnn.handler.train_cl import do_train_spike_class
            do_train_spike_class(DatasetLoader, dnn_handler, 'Config_Waveform', 'Waveforms')
        case 9:
            # --- Waveform Autoencoder
            from denspp.offline.dnn.handler.train_ae import do_train_neural_autoencoder
            do_train_neural_autoencoder(DatasetLoader, dnn_handler, 'Config_Waveform', 'Waveforms')
        case 10:
            # --- Neural Classifier (Normal)
            from denspp.offline.dnn.handler.train_cl import do_train_spike_class
            do_train_spike_class(DatasetLoader, dnn_handler, used_model_name='synthetic_cl_v1')
        case 11:
            # --- Autoencoder (Normal)
            from denspp.offline.dnn.handler.train_ae import do_train_neural_autoencoder
            do_train_neural_autoencoder(DatasetLoader, dnn_handler)
        case 12:
            # --- Autoencoder + Classifier
            from denspp.offline.dnn.handler.train_ae_cl import do_train_ae_classifier
            do_train_ae_classifier(DatasetLoader, dnn_handler)
        case 13:
            # --- Autoencoder + Classifier (Sweep Run of Hidden Layer Size)
            from denspp.offline.dnn.handler.train_ae_cl_sweep import do_train_ae_cl_sweep
            from denspp.offline.dnn.plots.plot_ae_cl_sweep import extract_data_from_files, plot_common_loss, plot_common_params, plot_architecture_metrics_isolated
            path2data = do_train_ae_cl_sweep(DatasetLoader, dnn_handler, 1, 1, 32)
            data = extract_data_from_files(path2data)
            plot_common_loss(data, path2save=path2data)
            plot_common_params(data, path2save=path2data)
            plot_architecture_metrics_isolated(data, show_plots=True, path2save=path2data)
        case 14:
            # --- Spike Detection
            from denspp.offline.dnn.handler.train_cl import do_train_spike_class
            do_train_spike_class(DatasetLoader, dnn_handler, 'Config_SDA', '', 'sda_dnn_v1')
        case _:
            raise NotImplementedError("Wrong model! Please select right model!")

    print("================================================================"
          "\nFinish!")
