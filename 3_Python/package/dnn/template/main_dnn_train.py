from package.dnn.dnn_handler import Config_ML_Pipeline, DefaultSettings_MLPipe
from package.yaml_handler import yaml_config_handler
from package.structure_builder import create_folder_general_firstrun


if __name__ == "__main__":
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)"
          "\n===========================================================================================")

    # --- Loading YAML-Settings file
    create_folder_general_firstrun()
    yaml_handler = yaml_config_handler(DefaultSettings_MLPipe, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(Config_ML_Pipeline)

    # --- Selecting model for train
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from package.dnn.handler.train_mnist import do_train_cl
            do_train_cl(dnn_handler)
        case 1:
            # --- MNIST (Autoencoder)
            from package.dnn.handler.train_mnist import do_train_ae
            do_train_ae(dnn_handler)
        case 2:
            # --- Autoencoder
            from package.dnn.handler.train_ae import do_train_neural_autoencoder
            do_train_neural_autoencoder(dnn_handler)
        case 3:
            # --- Classifier
            from package.dnn.handler.train_cl import do_train_neural_spike_classification
            do_train_neural_spike_classification(dnn_handler)
        case 4:
            # --- Autoencoder + Classifier
            from package.dnn.handler.train_aecl import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler)
        case _:
            raise NotImplementedError("Wrong model! Please select right model!")

    print("================================================================"
          "\nFinish!")
