from package.dnn.dnn_handler import dnn_handler, RecommendedDNNHandler
from package.yaml_handler import yaml_config_handler


if __name__ == "__main__":
    # --- Loading YAML-Settings file
    yaml_handler = yaml_config_handler(RecommendedDNNHandler, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(dnn_handler)

    # --- Selecting model for train
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
            # --- Autoencoder
            from package.dnn.template.handler.train_ae import do_train_ae
            do_train_ae(dnn_handler, 0, 0.01)
        case 3:
            # --- Classifier
            from package.dnn.template.handler.train_cl import do_train_classifier
            do_train_classifier(dnn_handler)
        case 4:
            # --- Autoencoder + Classifier
            from package.dnn.template.handler.train_ae_class import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler, 5, 5)
        case _:
            print("Wrong model! Please select right model!")
