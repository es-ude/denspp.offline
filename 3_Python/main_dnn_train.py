from package.dnn.dnn_handler import dnn_handler, RecommendedDNNHandler
from package.yaml_handler import yaml_config_handler


if __name__ == "__main__":
    # --- Loading YAML-Settings file
    yaml_handler = yaml_config_handler(RecommendedDNNHandler, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(dnn_handler)

    # --- Selecting model for train
    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)"
          "\n===========================================================================================")
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from package.dnn.template.handler.train_mnist import do_train_cl
            do_train_cl(dnn_handler.do_plot, dnn_handler.do_block)
        case 1:
            # --- MNIST (Autoencoder)
            from package.dnn.template.handler.train_mnist import do_train_ae
            do_train_ae(dnn_handler.do_plot, dnn_handler.do_block)
        case 2:
            # --- Autoencoder (Normal)
            from package.dnn.template.handler.train_ae import do_train_ae
            do_train_ae(dnn_handler)
        case 3:
            # --- Autoencoder + Classifier
            from package.dnn.template.handler.train_aecl import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler)
        case 4:
            # --- Spike Detection
            from src_dnn.train_sda import dnn_train_sda
            dnn_train_sda(dnn_handler)
        case 5:
            # --- RGC ON/OFF Classifier
            from src_dnn.train_rgc_class import do_train_rgc_class
            do_train_rgc_class(dnn_handler)
        case 6:
            # --- RGC ON/OFF Autoencoder + Classifier
            from src_dnn.train_rgc_ae_cl import do_train_rgc_ae_cl
            do_train_rgc_ae_cl(dnn_handler)
        case 7:
            # --- Neural Decoder (Utah Array)
            from src_dnn.train_decoder_utah import do_train_decoder_utah
            do_train_decoder_utah(dnn_handler)
        case _:
            print("Wrong model! Please select right model!")
    print("================================================================"
          "\nFinish!")
