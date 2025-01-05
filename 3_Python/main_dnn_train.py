from package.dnn.dnn_handler import ConfigMLPipeline, DefaultSettings_MLPipe
from package.yaml_handler import YamlConfigHandler
from package.structure_builder import create_folder_general_firstrun


if __name__ == "__main__":
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)"
          "\n===========================================================================================")

    create_folder_general_firstrun()
    # --- Loading YAML-Settings file
    yaml_handler = YamlConfigHandler(DefaultSettings_MLPipe, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(ConfigMLPipeline)

    # --- Processing
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from package.dnn.handler.train_mnist import do_train_mnist_cl
            do_train_mnist_cl(dnn_handler)
        case 1:
            # --- MNIST (Autoencoder)
            from package.dnn.handler.train_mnist import do_train_mnist_ae
            do_train_mnist_ae(dnn_handler)
        case 2:
            # --- Neural Classifier (Normal)
            from package.dnn.handler.train_cl import do_train_spike_class
            do_train_spike_class(dnn_handler, used_model_name='synthetic_cl_v1')
        case 3:
            # --- Autoencoder (Normal)
            from package.dnn.handler.train_ae import do_train_neural_autoencoder
            do_train_neural_autoencoder(dnn_handler)
        case 4:
            # --- Autoencoder + Classifier
            from package.dnn.handler.train_ae_cl import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler)
        case 5:
            # --- Autoencoder + Classifier (Sweep Run of Hidden Layer Size)
            from package.dnn.handler.train_ae_cl_sweep import do_train_ae_cl_sweep
            from package.dnn.plots.plot_ae_cl_sweep import extract_data_from_files, plot_common_loss, plot_common_params, plot_architecture_metrics_isolated
            path2data = do_train_ae_cl_sweep(dnn_handler, 1, 1, 32)
            data = extract_data_from_files(path2data)
            plot_common_loss(data, path2save=path2data)
            plot_common_params(data, path2save=path2data)
            plot_architecture_metrics_isolated(data, show_plots=True, path2save=path2data)
        case 6:
            # --- Spike Detection
            from package.dnn.handler.train_cl import do_train_spike_class
            do_train_spike_class(dnn_handler, 'Config_SDA', '', 'sda_dnn_v1')
        case 7:
            # --- RGC ON/OFF Classifier
            from src_dnn.train_rgc import do_train_rgc_class
            do_train_rgc_class(dnn_handler)
        case 8:
            # --- RGC ON/OFF Autoencoder + Classifier
            from src_dnn.train_rgc import do_train_rgc_aecl
            do_train_rgc_aecl(dnn_handler)
        case 9:
            # --- Neural Decoder (Utah Array)
            from src_dnn.train_decoder_utah import do_train_decoder_utah
            do_train_decoder_utah(dnn_handler)
        case _:
            raise NotImplementedError("Wrong model! Please select right model!")

    print("================================================================"
          "\nFinish!")
