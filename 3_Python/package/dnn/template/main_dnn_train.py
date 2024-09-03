from package.dnn.dnn_handler import dnn_handler

dnn_handler = dnn_handler(
    mode_dnn=0,
    mode_cellbib=0,
    do_plot=True,
    do_block=True
)


if __name__ == "__main__":
    # --- Configs (AE)
    mode_ae = 0
    noise_std_ae = 0.01

    # --- Selecting model for train
    match dnn_handler.mode_train_dnn:
        case 0:
            # --- MNIST (Classifier)
            from src_dnn.train_mnist import do_train_cl
            do_train_cl(dnn_handler.do_plot, dnn_handler.do_block)
        case 1:
            # --- MNIST (Autoencoder)
            from src_dnn.train_mnist import do_train_ae
            do_train_ae(dnn_handler.do_plot, dnn_handler.do_block)
        case 2:
            # --- Autoencoder
            from src_dnn.train_ae import do_train_ae
            do_train_ae(dnn_handler, mode_ae, noise_std_ae)
        case 3:
            # --- Classifier
            from src_dnn.train_cl import do_train_cl
            do_train_cl(dnn_handler)
        case 4:
            # --- Autoencoder + Classifier
            from src_dnn.train_ae_class import do_train_ae_classifier
            do_train_ae_classifier(dnn_handler, 5, 5)
        case _:
            print("Wrong model! Please select right model!")
